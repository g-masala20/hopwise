# @Time   : 2025
# @Author : Giacomo Medda, Alessandro Soccol
# @Email  : giacomo.medda@unica.it, alessandro.soccol@unica.it

"""hopwise.model.logits_processor
#############################
Common logits processor in recommender system
"""

import inspect

import numpy as np
import torch
from cachetools import LFUCache
import math
from hopwise.utils import KnowledgeEvaluationType
import random
from hopwise.utils import PathLanguageModelingTokenType

import graphviz # AGGIUNTA PER LA STAMPA DEL GRAFO (che io sia maledetto)
import pdb; #pdb.set_trace() # AGGIUNTA PER IL DEBUG (perchè ogni tanto da problemi .-.)
import os
import hashlib
from concurrent.futures import ThreadPoolExecutor

class LogitsProcessor:
    """
    Abstract base class for all logit processors that can be applied during generation.
    Copy of HuggingFace's LogitsProcessor.
    """

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class LogitsProcessorList(list):
    """
    This class can be used to create a list of [`LogitsProcessor`] to subsequently process a `scores` input tensor.
    This class inherits from list and adds a specific *__call__* method to apply each [`LogitsProcessor`] to the
    inputs.
    Copy of HuggingFace's LogitsProcessorList.
    """

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
            scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
                Prediction scores of a language modeling head. These can be logits for each vocabulary when not using
                beam search or log softmax for each vocabulary token when using beam search
            kwargs (`Dict[str, Any]`, *optional*):
                Additional kwargs that are specific to a logits processor.

        Return:
            `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`:
                The processed prediction scores.

        """
        for processor in self:
            function_args = inspect.signature(processor.__call__).parameters
            if len(function_args) > 2:  # noqa: PLR2004
                if not all(arg in kwargs for arg in list(function_args.keys())[2:]):
                    raise ValueError(
                        f"Make sure that all the required parameters: {list(function_args.keys())} for "
                        f"{processor.__class__} are passed to the logits processor."
                    )
                scores = processor(input_ids, scores, **kwargs)
            else:
                scores = processor(input_ids, scores)

        return scores


class ConstrainedLogitsProcessorWordLevel(LogitsProcessor):
    """
    Force the last token to be one of the force_tokens if the total length is reached, in the path generation stage
    this means to limit the hop size. This is a word-level constraint, does not work with piece tokenizers.
    If task is link prediction (LP) logit processor forces last token to reachable ones
    """

    def __init__(
        self,
        tokenized_ckg,
        tokenized_used_ids,
        max_sequence_length,
        tokenizer,
        mask_cache_size=3 * 10**4,
        pos_candidates_cache_size=1 * 10**5,
        task=KnowledgeEvaluationType.REC,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tokenized_ckg = tokenized_ckg
        self.tokenized_used_ids = tokenized_used_ids
        self.max_sequence_length = max_sequence_length
        self.tokenizer = tokenizer
        self.bos_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)
        self.pos_candidates_cache = LFUCache(pos_candidates_cache_size)
        self.mask_cache = LFUCache(mask_cache_size)
        self.task = task

        if self.task == KnowledgeEvaluationType.LP:
            self.special_tokens_ids = [
                self.tokenizer.encode(x, add_special_tokens=False)[0]
                for x in self.tokenizer.all_special_tokens_extended
            ]
        else:
            self.special_tokens_ids = None

    def is_bos_token_in_input(self, input_ids):
        """Check if the input contains a BOS token. Checking the first sequence is enough."""
        return (input_ids[0, 0] == self.bos_token_id).item()

    def __call__(self, input_ids, scores):
        current_len = input_ids.shape[-1]
        has_bos_token = self.is_bos_token_in_input(input_ids)

        unique_input_ids = input_ids
        if self.task == KnowledgeEvaluationType.REC and current_len < self.max_sequence_length - 1 - has_bos_token:
            # Determine whether the next token to generate is a relation or an entity:
            # - relation: only the last entity is needed (1 token) → for [user123] last_n_tokens = 1
            # - entity: the last 2 tokens are needed (entity, relation) → for [user123, watched] last_n_tokens = 2
            # Apply deduplication: select unique sequences based only on the relevant last tokens (1 or 2)
            # This avoids recomputing the same mask for sequences that share the same context
            last_n_tokens = 2 if self.is_next_token_entity(input_ids) else 1
            _, input_ids_indices, input_ids_inv = np.unique(
                input_ids.cpu().numpy()[:, -last_n_tokens:], axis=0, return_index=True, return_inverse=True
            )
            unique_input_ids = input_ids[input_ids_indices]

        full_mask = np.zeros((unique_input_ids.shape[0], len(self.tokenizer)), dtype=bool)
        for idx in range(unique_input_ids.shape[0]):
            if self.task == KnowledgeEvaluationType.REC:
                key, candidate_tokens = self.process_scores_rec(unique_input_ids, idx)
            elif self.task == KnowledgeEvaluationType.LP:
                key, candidate_tokens = self.process_scores_lp(unique_input_ids, idx)

            banned_mask = self.get_banned_mask(key, candidate_tokens)

            if banned_mask.all():
                banned_mask[self.tokenizer.pad_token_id] = False

            full_mask[idx] = banned_mask

        if self.task == KnowledgeEvaluationType.REC and current_len < self.max_sequence_length - 1 - has_bos_token:
            scores[full_mask[input_ids_inv]] = -torch.inf
        else:
            scores[full_mask] = -torch.inf

        return scores

    def process_scores_rec(self, input_ids, idx):
        """Process each score based on input length and update mask list."""
        current_len = input_ids.shape[-1]
        has_bos_token = self.is_bos_token_in_input(input_ids)

        key = self.get_current_key(input_ids, idx)
        if current_len == self.max_sequence_length - 1 - has_bos_token:
            current_uid = input_ids[idx, int(has_bos_token)].item()
            uid_cond_key = (current_uid, *key)

            candidate_tokens = self.pos_candidates_cache.get(uid_cond_key)
            if candidate_tokens is None:
                candidate_tokens = self.get_candidates_rec(*key)

                # Get user positives
                user_used_ids = self.tokenized_used_ids[current_uid]
                # Select negatives
                candidate_tokens = list(candidate_tokens - user_used_ids)
                # Useless if during evaluation a user is seen once
                self.pos_candidates_cache[uid_cond_key] = candidate_tokens
        else:
            candidate_tokens = list(self.get_candidates_rec(*key))

        return key, candidate_tokens

    def process_scores_lp(self, input_ids, idx):
        """Process each score based on input length or skip."""
        current_len = input_ids.shape[-1]
        has_bos_token = self.is_bos_token_in_input(input_ids)

        key, candidate_tokens = None, None
        if current_len % 2 == has_bos_token:
            key = self.get_current_key(input_ids, idx)
            candidate_tokens = self.get_candidates_lp(key)

        return key, candidate_tokens

    def is_next_token_entity(self, input_ids):
        current_len = input_ids.shape[-1]
        has_bos_token = self.is_bos_token_in_input(input_ids)

        # bos_token determines if the current length is even or odd
        return current_len % 2 == has_bos_token

    def get_current_key(self, input_ids, idx):
        if self.is_next_token_entity(input_ids):
            return input_ids[idx, -2].item(), input_ids[idx, -1].item()
        else:
            # The next token is a relation
            return (input_ids[idx, -1].item(),)

    def get_candidates_rec(self, key1, key2=None):
        """
        :param key1:
        :param key2: if key2 is not None, it returns entity candidates, otherwise relation candidates
        """
        if key1 in self.tokenized_ckg:
            if key2 is not None and key2 in self.tokenized_ckg[key1]:
                # return tail given head + relation
                return self.tokenized_ckg[key1][key2]
            else:
                # return relations given head
                return set(self.tokenized_ckg[key1].keys())
        else:
            raise ValueError(f"Key {key1} ('{self.tokenizer.convert_ids_to_tokens(key1)}') not found in tokenized_ckg")

    def get_candidates_lp(self, key):
        return list(self.tokenized_used_ids[key]) + self.special_tokens_ids

    def get_banned_mask(self, key, candidate_tokens):
        """Retrieve or cache the banned token mask for a specific key."""
        banned_mask = self.mask_cache.get(key)
        if banned_mask is None:
            banned_mask = np.ones(len(self.tokenizer), dtype=bool)
            banned_mask[candidate_tokens] = False
            self.mask_cache[key] = banned_mask
        return banned_mask



class ConstrainedLogitsProcessorWordLevelDevel(ConstrainedLogitsProcessorWordLevel):
    
    """
    Force the last token to be one of the force_tokens if the total length is reached, in the path generation stage
    this means to limit the hop size. This is a word-level constraint, does not work with piece tokenizers.
    If task is link prediction (LP) logit processor forces last token to reachable ones
    """


    def __init__(
        self,
        tokenized_ckg,
        tokenized_used_ids,
        max_sequence_length,
        tokenizer,
        train_dataset,  # aggiunta mia 
        mask_cache_size=3 * 10**4,
        pos_candidates_cache_size=1 * 10**5,
        task=KnowledgeEvaluationType.REC,
        **kwargs,
    ):
        super().__init__(
        tokenized_ckg,
        tokenized_used_ids,
        max_sequence_length,
        tokenizer,
        mask_cache_size,
        pos_candidates_cache_size,
        task,
        **kwargs)
        self.train_dataset = train_dataset
        self._graph_generated = False  # Flag per controllare se il grafo è già stato generato

    def __call__(self, input_ids, scores):

        train_dataset = self.train_dataset
        entity_mapping = train_dataset.field2token_id['entity_id']
        #print("DEBUG: ConstrainedLogitsProcessorWordLevelDevel __call__ invoked AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa")

        #breakpoint()
        current_len = input_ids.shape[-1]
        has_bos_token = self.is_bos_token_in_input(input_ids)

        unique_input_ids = input_ids
        if self.task == KnowledgeEvaluationType.REC and current_len < self.max_sequence_length - 1 - has_bos_token:
            user_idx = 1
            
            _, input_ids_indices, input_ids_inv = np.unique(
                input_ids.cpu().numpy()[:, [user_idx]], axis=0, return_index=True, return_inverse=True
            )
            unique_input_ids = input_ids[input_ids_indices]

        # ---

        # Estrai i constraint dal dataset (con fallback se non esiste)

        constraints_tokens_map = train_dataset.field2id_token['constraints']
        constraints_tokens = train_dataset.get_user_feature().interaction["constraints"]

        def usertoken_tokenizer2id(_id):
            return int(self.tokenizer.convert_ids_to_tokens(_id)[1:])
    
        #user_feat = train_dataset.get_user_feature()   

        #breakpoint()

        try:
            #constraints_col = user_feat.interaction.get('constraints', None)
            
            # Solo se i constraints esistono, procedi con il sistema di restrizioni
            hard_restriction_keys_per_user = [[] for _ in range(unique_input_ids.shape[0])]
            soft_restriction_keys_per_user = [[] for _ in range(unique_input_ids.shape[0])]
            preferences_keys_per_user = [[] for _ in range(unique_input_ids.shape[0])]

            #breakpoint()

            for idx in range(unique_input_ids.shape[0]):
                user_position = 1 if has_bos_token else 0
                user_idx_in_batch = unique_input_ids[idx, user_position].item()

                #print(f"DEBUG: Processing user_idx_in_batch={user_idx_in_batch}")

                remapped_user_idx_in_batch = usertoken_tokenizer2id(user_idx_in_batch)
                user_constraints_ids = constraints_tokens[remapped_user_idx_in_batch]
                user_constraints_tokens = constraints_tokens_map[user_constraints_ids].split(',')
                user_constraints_ids = [entity_mapping[constraint] for constraint in user_constraints_tokens]

                #debug print
                #print(f"DEBUG: user_idx_in_batch={user_idx_in_batch}, user_constraints_tokens={user_constraints_tokens}")

                # TODO: potrei far diventare tutto una funzione? boh magari dopo se serve
                if user_constraints_tokens != ['']:
                    
                    if isinstance(user_constraints_tokens, str):
                        user_constraints_list = [user_constraints_tokens]
                    else:
                        user_constraints_list = list(user_constraints_tokens)

                    hard_constraints = user_constraints_list[:2] if len(user_constraints_list) >= 2 else [] # Primi due constraint per hard restrictions
                    soft_constraints = user_constraints_list[2:4] if len(user_constraints_list) >= 4 else [] # Terzo e quarto constraint per soft restrictions 
                    preferences = user_constraints_list[4:] if len(user_constraints_list) > 4 else [] # Resto dei constraint come preferenze (in teoria la quinta e sesta)
                    
                    hard_restriction_keys_per_user[idx].extend(hard_constraints)
                    soft_restriction_keys_per_user[idx].extend(soft_constraints)
                    preferences_keys_per_user[idx].extend(preferences)
                else:
                    # Fallback per utenti senza constraints
                    # WARNING: Constraints not available in dataset: 'NoneType' object has no attribute 'append'
                    hard_restriction_keys_per_user[idx].extend([])
                    soft_restriction_keys_per_user[idx].extend([])
                    preferences_keys_per_user[idx].extend([])


            #breakpoint()
            
        except (KeyError, AttributeError) as e:
            # Se il dataset non ha constraints, usa liste vuote per tutti gli utenti
            print(f"WARNING: Constraints not available in dataset: {e}")
            constraints_tokens = None
            hard_restriction_keys_per_user = [[] for _ in range(unique_input_ids.shape[0])]
            soft_restriction_keys_per_user = [[] for _ in range(unique_input_ids.shape[0])]
            preferences_keys_per_user = [[] for _ in range(unique_input_ids.shape[0])]

        # ---

        # STAMPA GRAFO DEL PRIMO UTENTE (prima DELLE restrizioni)
        if not self._graph_generated:
            # Grafo senza restrizioni
            self._debug_visualize_first_user(hard_restriction_keys_per_user[0], 
                                                soft_restriction_keys_per_user[0], 
                                                preferences_keys_per_user[0], 
                                                train_dataset)
            # Grafo con hard restrictions evidenziate
            #self._debug_visualize_first_user_with_hard_restrictions(hard_restriction_keys_per_user[0],
            #                                                        soft_restriction_keys_per_user[0],
            #                                                        preferences_keys_per_user[0],
            #                                                        train_dataset)
            self._graph_generated = True

        full_mask = np.zeros((unique_input_ids.shape[0], len(self.tokenizer)), dtype=bool) # Maschera completa inizializzata a zero, dimensioni)
        #breakpoint()

        # ---

        for idx in range(unique_input_ids.shape[0]):
            if self.task == KnowledgeEvaluationType.REC:
                key, candidate_tokens = self.process_scores_rec(unique_input_ids, idx)
            elif self.task == KnowledgeEvaluationType.LP:
                key, candidate_tokens = self.process_scores_lp(unique_input_ids, idx)
            
            # --- INIZIO MASKING:

            # Inizializza la maschera base una sola volta
            banned_mask = self.get_banned_mask(key, candidate_tokens)
            full_mask[idx] = banned_mask.copy()
            
            # DEBUG: Controlla se già mascherato dalla logica base
            if np.all(full_mask[idx]):
                #breakpoint()
                print(f"DEBUG: Base mask already blocks all tokens for idx={idx} (BEFORE constraints), this is a problem!")

            # --- HARD MASKING:

            # Applica hard restrictions solo se i constraints esistono
            if constraints_tokens is not None and hard_restriction_keys_per_user[idx]!=[]:

                hard_restriction_mask = np.zeros_like(full_mask[idx], dtype=bool)
                
                # Accumula tutte le restrizioni hard per questo utente
                for hard_key in hard_restriction_keys_per_user[idx]:
                    constraint_mask = self.gen_mask_from_key(hard_key, train_dataset, mask_type="ban")
                    hard_restriction_mask = np.logical_or(hard_restriction_mask, constraint_mask)

                    if np.all(full_mask[idx]): 
                        print(f"DEBUG: All tokens masked for idx={idx} (h_constraints cycles)")
                        raise RuntimeError("All tokens are masked for all input rows in full_mask.(h_constraints cycles)")

                # STAMPA GRAFO DEL PRIMO UTENTE (dopo le hard restrictions)
                if idx == 0:
                    self._debug_visualize_first_user_with_hard_restrictions(hard_restriction_keys_per_user[idx], soft_restriction_keys_per_user[idx], preferences_keys_per_user[idx], train_dataset, full_mask[idx])

                
                # Combina maschera base con restrizioni hard
                full_mask[idx] = np.logical_or(banned_mask, hard_restriction_mask)

            if np.all(full_mask[idx]):    
                print(f"DEBUG: All tokens masked for idx={idx} (AFTER constraints)")
                raise RuntimeError("All tokens are masked for all input rows in full_mask.(AFTER constraints)")

            
            breakpoint()
            # --- SOFT MASKING:

            # Applica soft masking solo se i constraints esistono
            if constraints_tokens is not None and soft_restriction_keys_per_user[idx]!=[]:
                # sintesi: vengono ordinati i valori in modo decrescente in base al numero di token connessi,
                # viene applicata la maschera per volta, e si applica la successiva solo se il kg è valido,
                # altrimenti si interrompe il processo 
                
                # Solo ordinare se ci sono effettivamente constraint soft
                if soft_restriction_keys_per_user[idx]:
                    soft_restriction_keys_per_user[idx] = sorted(
                        soft_restriction_keys_per_user[idx],
                        key=lambda k: len(self.tokenized_ckg[entity_mapping[k]]) if k in entity_mapping and entity_mapping[k] in self.tokenized_ckg else 0,
                        reverse=True,
                    )

                soft_mask = np.zeros_like(full_mask[idx], dtype=bool) # necessario? non credo

                for soft_key in soft_restriction_keys_per_user[idx]:
                    #if soft_key in entity_mapping:
                    soft_mask = np.logical_or(full_mask[idx], self.gen_mask_from_key(soft_key, train_dataset, mask_type="ban"))
                    if np.all(soft_mask): 
                        break
                    else:
                        full_mask[idx] = soft_mask

            # --- PREFERENCE MASKING:
            # (TODO: proof of concept, probabilmente da rivedere)
            if constraints_tokens is not None and preferences_keys_per_user[idx]!=[]:
                preference_mask = np.ones_like(full_mask[idx], dtype=bool)  # Maschera inizializzata a True (tutti i nodi bannati)

                # Applica le preferenze per l'utente corrente
                for pref_key in preferences_keys_per_user[idx]:
                    # Abilita il nodo e i suoi vicini   
                    preference_mask = np.logical_or(full_mask[idx], self.gen_mask_from_key(pref_key, train_dataset, mask_type="allow")) 
                    #TODO: assicurarsi che non crei sottografi separati e che tutte le preference siano rispettate allo stesso tempo, non ho la testa ora :(
                    if np.all(preference_mask): 
                        break
                    else:
                        full_mask[idx] = preference_mask

                # Combina la maschera di preferenza con quella generale (Ma solo se non blocca tutto)
                #if not np.all(preference_mask):
                #    full_mask[idx] = np.logical_and(full_mask[idx], preference_mask)

                # DEBUG: Controlla se la maschera blocca tutto
                if np.all(full_mask[idx]):
                    print(f"DEBUG: All tokens masked for idx={idx} (AFTER preferences)")
                    raise RuntimeError("All tokens are masked for all input rows in full_mask (AFTER preferences).")
    
        #---

        if self.task == KnowledgeEvaluationType.REC and current_len < self.max_sequence_length - 1 - has_bos_token:
            scores[full_mask[input_ids_inv]] = -math.inf
        else:
            scores[full_mask] = -math.inf

        return scores
    
    def generate_and_save_constraints(self, user_file_path, n_constraints=6, output_file_path=None):
        """
        Aggiorna il file utente aggiungendo la colonna 'constraints',
        per ogni utente genera n_constraints restrizioni casuali da entity_mapping.
        Se output_file_path è None, sovrascrive user_file_path, altrimenti salva su output_file_path.

        realisticamente, non dovrebbe finire nella codebase, ma per il momento va bene così.
        """
        import random
        
        entity_mapping = self.train_dataset.field2token_id['entity_id']
        all_entity_keys = list(entity_mapping.keys())

        with open(user_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        header = lines[0].strip().split()
        if "constraints" not in header:
            header.append("constraints")

        new_lines = [header]
        for line in lines[1:]:
            fields = line.strip().split()
            constraints = ",".join(random.sample(all_entity_keys, n_constraints))
            if len(fields) < len(header):
                fields.append(constraints)
            else:
                fields[header.index("constraints")] = constraints
            new_lines.append(fields)

        output_path = output_file_path if output_file_path is not None else user_file_path
        with open(output_path, "w", encoding="utf-8") as f:
            for row in new_lines:
                f.write("\t".join(row) + "\n")

    def extract_connected_entities(self, token_id):
        """
        Estrae tutte le entità connesse al token_id, sotto forma di lista completamente appiattita.

        Args:
            token_id (int): L'ID del token per cui estrarre le entità connesse.

        Returns:
            list: Lista di entità connesse.
        """

        connected_entities = set()
        for entity_set in self.tokenized_ckg[token_id].values():
            connected_entities.update(entity_set if isinstance(entity_set, (list, set)) else [entity_set])
        connected_entities = list(set(connected_entities))#.sorted()  # l'ordine è temporaneo, serve solo a render più leggibile l'output
        
        return connected_entities
    
    def gen_mask_from_key(self, key, train_dataset, include_connected_entities=True, mask_type="ban"):
        """
        Genera una maschera per un dato nodo (key) e opzionalmente per i nodi connessi.

        Args:
            key (str): La chiave dell'entità per cui generare la maschera.
            train_dataset: Il dataset di allenamento contenente il mapping delle entità.
            include_connected_entities (bool): Se True, include i nodi connessi nella maschera.
            mask_type (str): Tipo di maschera da generare:
                - "ban": Banna i nodi specificati (restrizioni).
                - "allow": Abilita i nodi specificati (preferenze).

        Returns:
            np.ndarray: Maschera booleana generata.
        """

        mask_type = mask_type.lower().strip()

        mask = np.zeros(len(self.tokenizer), dtype=bool) if mask_type == "ban" else np.ones(len(self.tokenizer), dtype=bool)

        # Ottieni l'ID interno del nodo
        id_interno = train_dataset.field2token_id['entity_id'][key]
        if id_interno < train_dataset.item_num:
            token = [PathLanguageModelingTokenType.ITEM.value[0]] + str(id_interno)
        else:
            token = PathLanguageModelingTokenType.ENTITY.value[0] + str(id_interno)

        token_id = self.tokenizer.convert_tokens_to_ids(token)
        mask[token_id] = True if mask_type == "ban" else False

        # Gestisci i nodi connessi
        if include_connected_entities and token_id in self.tokenized_ckg:
            connected_entities = self.extract_connected_entities(token_id)
            mask[connected_entities] = True if mask_type == "ban" else False

        return mask

    def _debug_visualize_first_user(self, hard_keys, soft_keys, pref_keys, train_dataset):
        """Genera e salva il grafo del primo utente prima delle restrizioni."""
        
        print(f"DEBUG: _debug_visualize_first_user called")
        print(f"DEBUG: hard_keys = {hard_keys}")
        print(f"DEBUG: soft_keys = {soft_keys}")
        print(f"DEBUG: pref_keys = {pref_keys}")

        #breakpoint()
        
        # per evitare un crash se sono vuoti, si potrebbe fare meglio ma finchè funziona va bene
        if not hard_keys or not hard_keys[0]:
            return
        first_user_constraints = hard_keys[0] + soft_keys[0] + pref_keys[0]
        if not first_user_constraints:
            return
        
        # Converti le chiavi in token_ids
        entity_mapping = train_dataset.field2token_id['entity_id']
        constraint_token_ids = []
        for key in first_user_constraints:
            if key in entity_mapping:
                id_interno = entity_mapping[key]
                if id_interno < train_dataset.item_num:
                    token = PathLanguageModelingTokenType.ITEM.value[0] + str(id_interno)
                else:
                    token = PathLanguageModelingTokenType.ENTITY.value[0] + str(id_interno)
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                constraint_token_ids.append(token_id)
        
        # Genera il grafo
        if constraint_token_ids:
            subgraph = ConstrainedLogitsProcessorWordLevelDevel.extract_subgraph(self.tokenized_ckg, constraint_token_ids, depth=1) # TODO: DEPTH 2! al momento è a 1 per fare prove
            
            ConstrainedLogitsProcessorWordLevelDevel.convert_ckg_to_graphviz(
                subgraph, 
                output_file="debug_first_user_before_restrictions", 
                cache=False, 
                tokenizer=self.tokenizer
            )
            print(f"DEBUG: Grafo del primo utente salvato (prima delle restrizioni)")

    def _debug_visualize_first_user_with_hard_restrictions(self, hard_keys, soft_keys, pref_keys, train_dataset, hard_restriction_mask):
        """
        Genera e salva il grafo del primo utente mostrando in rosso i nodi bannati dalle hard restrictions.
        
        Args:
            hard_keys: Lista di chiavi per hard restrictions
            soft_keys: Lista di chiavi per soft restrictions
            pref_keys: Lista di chiavi per preferences
            train_dataset: Dataset di training contenente i mapping
        """
        print(f"DEBUG: _debug_visualize_first_user_with_hard_restrictions called")
        print(f"DEBUG: hard_keys = {hard_keys}")
        print(f"DEBUG: soft_keys = {soft_keys}")
        print(f"DEBUG: pref_keys = {pref_keys}")
        
        # Validazione input
        if not hard_keys or not hard_keys[0]:
            print("DEBUG: Nessuna hard restriction da visualizzare")
            return
        
        first_user_constraints = hard_keys[0] + soft_keys[0] + pref_keys[0]
        if not first_user_constraints:
            return
        
        # Converti le chiavi in token_ids
        entity_mapping = train_dataset.field2token_id['entity_id']
        constraint_token_ids = []
        for key in first_user_constraints:
            if key in entity_mapping:
                id_interno = entity_mapping[key]
                if id_interno < train_dataset.item_num:
                    token = PathLanguageModelingTokenType.ITEM.value[0] + str(id_interno)
                else:
                    token = PathLanguageModelingTokenType.ENTITY.value[0] + str(id_interno)
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                constraint_token_ids.append(token_id)
        
        if not constraint_token_ids:
            return
        
        # Estrai il sottografo
        subgraph = ConstrainedLogitsProcessorWordLevelDevel.extract_subgraph(
            self.tokenized_ckg, constraint_token_ids, depth=1
        )
        
        # Converti il grafo in Graphviz applicando la maschera
        ConstrainedLogitsProcessorWordLevelDevel.convert_ckg_to_graphviz(
            subgraph, 
            output_file="debug_first_user_with_hard_restrictions",
            mask=hard_restriction_mask,
            tokenizer=self.tokenizer,
            cache=False
        )
        
        print(f"DEBUG: Grafo con hard restrictions salvato come debug_first_user_with_hard_restrictions.svg")
        print(f"DEBUG: Nodi bannati (rossi): {np.sum(hard_restriction_mask)} su {len(self.tokenizer)} totali")

# --- inizio funzioni graphviz

    def extract_subgraph(tokenized_ckg, focus_nodes, depth=1):
        """
        Estrae un sottografo contenente i nodi specificati e i loro vicini fino a una certa profondità.

        Args:
            tokenized_ckg (dict): Dizionario che rappresenta il grafo completo.
            focus_nodes (list): Lista di nodi da includere nel sottografo.
            depth (int): Profondità della ricerca

        Returns:
            dict: Sottografo contenente i nodi specificati e i loro vicini.
        """
        nodes_to_include = set(focus_nodes)
        current_level = set(focus_nodes)

        for _ in range(depth):
            next_level = set()
            for node in current_level:
                if node in tokenized_ckg:
                    for relation, connected_nodes in tokenized_ckg[node].items():
                        next_level.update(connected_nodes)
            nodes_to_include.update(next_level)
            current_level = next_level

        # Costruisci il sottografo
        subgraph = {}
        for node in nodes_to_include:
            if node in tokenized_ckg:
                subgraph[node] = {}
                for relation, connected_nodes in tokenized_ckg[node].items():
                    filtered_nodes = [n for n in connected_nodes if n in nodes_to_include]
                    if filtered_nodes:
                        subgraph[node][relation] = filtered_nodes

        return subgraph

    def add_edges(dot, node, edges, subgraph_nodes=None):
        """Aggiunge nodi e archi al grafo, filtrando archi verso nodi non presenti nel sottografo."""
        dot.node(str(node), label=str(node))
        for relation, connected_nodes in edges.items():
            for connected_node in connected_nodes:
                # Disegna l'arco solo se il nodo connesso è nel sottografo
                if subgraph_nodes is None or connected_node in subgraph_nodes:
                    dot.edge(str(node), str(connected_node), label=str(relation))

    def update_node_colors(dot, mask, tokenizer, banned_color="", allowed_color=""):
        """
        Aggiorna i colori dei nodi nel grafo in base a una maschera booleana.

        Se banned_color o allowed_color è una stringa vuota (o valore falsy), non
        modifica il colore dei nodi di quel tipo.
        """
        
        for token_id, is_banned in enumerate(mask):
            node_name = str(tokenizer.convert_ids_to_tokens(token_id))
            if is_banned:
                if banned_color:  # se vuoto -> non toccare i nodi bannati
                    dot.node(node_name, style="filled", fillcolor=banned_color)
            else:
                if allowed_color:  # se vuoto -> non toccare i nodi permessi
                    dot.node(node_name, style="filled", fillcolor=allowed_color)

    def convert_ckg_to_graphviz(tokenized_ckg, output_file="graph", mask=None, tokenizer=None, cache=True):
        """
        Converte il tokenized_ckg in un grafo stampabile con Graphviz, con supporto per caching, parallelizzazione e aggiornamento dei colori.

        Args:
            tokenized_ckg (dict): Dizionario che rappresenta il grafo.
            output_file (str): Nome del file di output (senza estensione).
            mask (np.ndarray): Maschera booleana (True = bannato, False = permesso).
            tokenizer: Tokenizer per convertire gli ID dei token in stringhe.
            cache (bool): Se True, usa il file di cache se esiste (disabilitato se mask è presente).

        Returns:
            graphviz.Digraph: Oggetto Graphviz del grafo.
        """
        # Disabilita la cache se c'è una maschera (perché la maschera cambia i colori)
        use_cache = cache and mask is None
        
        # Calcola l'hash della struttura del grafo
        ckg_str = str(tokenized_ckg).encode('utf-8')
        graph_hash =  hashlib.md5(ckg_str).hexdigest()
        cache_file = f"{output_file}_{graph_hash}.gv"

        # Usa il file di cache per la struttura del grafo se esiste
        if use_cache and os.path.exists(cache_file):
            print(f"DEBUG: Usando la struttura del grafo in cache {cache_file}")
            dot = graphviz.Source.from_file(cache_file)
        else:
            # Crea un nuovo grafo
            dot = graphviz.Digraph(format="svg")

            # Ottieni l'insieme dei nodi del sottografo per il filtro
            subgraph_nodes = set(tokenized_ckg.keys())

            # Parallelizza l'aggiunta di nodi e archi
            with ThreadPoolExecutor() as executor:
                futures = []
                for node, edges in tokenized_ckg.items():
                    futures.append(executor.submit(ConstrainedLogitsProcessorWordLevelDevel.add_edges, dot, node, edges, subgraph_nodes))
                for future in futures:
                    future.result()

            # Salva la struttura del grafo in cache solo se non c'è la maschera
            if use_cache:
                dot.save(cache_file)
                print(f"DEBUG: Struttura del grafo salvata in {cache_file}")

        # Aggiorna i colori dei nodi in base alla maschera
        if mask is not None and tokenizer is not None:
            ConstrainedLogitsProcessorWordLevelDevel.update_node_colors(dot, mask, tokenizer, banned_color="red", allowed_color="lightgreen")

        # Salva il grafo su file
        try:
            dot.render(output_file, cleanup=True)
            print(f"DEBUG: Grafo salvato come {output_file}.svg")
        except Exception as e:
            print(f"DEBUG: Errore durante il rendering del grafo: {e}")

        return dot



class PrefixConstrainedLogitsProcessorWordLevel(ConstrainedLogitsProcessorWordLevel):
    def __init__(
        self,
        tokenized_ckg,
        tokenized_used_ids,
        max_sequence_length,
        tokenizer,
        **kwargs,
    ):
        super().__init__(
            tokenized_ckg,
            tokenized_used_ids,
            max_sequence_length,
            tokenizer,
            **kwargs,
        )
        self.mask_cache = None

    def __call__(self, input_ids, scores):
        current_len = input_ids.shape[-1]
        if current_len == self.max_sequence_length - 1:
            self.mask_non_eos_tokens(scores)
        else:
            indices = []
            masked_scores = torch.full_like(scores, -torch.inf)
            for idx in range(scores.shape[0]):
                _, candidate_tokens = self.process_scores(input_ids, idx, current_len)

                candidate_tokens = torch.LongTensor(candidate_tokens, device=scores.device)
                indices.append(candidate_tokens)
                masked_scores[idx].scatter_(dim=-1, index=candidate_tokens, src=scores[idx])
            scores = masked_scores

        return scores


class PLMLogitsProcessorWordLevel(LogitsProcessor):
    """
    https://dl.acm.org/doi/pdf/10.1145/3485447.3511937
    Constraint decoding strategy for PLM, it forces the model to generate alternatively entities and relations
    """

    def __init__(
        self,
        tokenized_ckg,
        tokenized_used_ids,
        max_sequence_length,
        tokenizer,
        pos_candidates_cache_size=1 * 10**5,
        task=KnowledgeEvaluationType.REC,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tokenized_ckg = tokenized_ckg
        self.tokenized_used_ids = tokenized_used_ids
        self.max_sequence_length = max_sequence_length
        self.tokenizer = tokenizer
        self.bos_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)
        self.pos_candidates_cache = LFUCache(pos_candidates_cache_size)
        self.task = task

        if self.task == KnowledgeEvaluationType.LP:
            self.special_tokens_ids = [
                self.tokenizer.encode(x, add_special_tokens=False)[0]
                for x in self.tokenizer.all_special_tokens_extended
            ]
        else:
            self.special_tokens_ids = None

        self.entity_token_ids = torch.LongTensor(list(set(self.tokenized_ckg.keys())))
        self.relation_token_ids = torch.LongTensor(
            list(set([rel for rel_dict in self.tokenized_ckg.values() for rel in rel_dict.keys()]))
        )

    def __call__(self, input_ids, scores):
        current_len = input_ids.shape[-1]
        has_bos_token = self.is_bos_token_in_input(input_ids)

        unique_input_ids = input_ids
        if self.task == KnowledgeEvaluationType.REC and current_len == (self.max_sequence_length - 1 - has_bos_token):
            user_idx = int(has_bos_token)
            _, input_ids_indices, input_ids_inv = np.unique(
                input_ids.cpu().numpy()[:, [user_idx]], axis=0, return_index=True, return_inverse=True
            )
            unique_input_ids = input_ids[input_ids_indices]

            full_mask = np.ones((unique_input_ids.shape[0], len(self.tokenizer)), dtype=bool)
            for idx in range(unique_input_ids.shape[0]):
                candidate_tokens = self.process_scores(unique_input_ids, idx)
                full_mask[idx, candidate_tokens] = False

            scores[full_mask[input_ids_inv]] = -torch.inf
        else:
            # Paths are expected to be the same type and length, so we can use the same mask for all
            full_mask = np.ones((unique_input_ids.shape[0], len(self.tokenizer)), dtype=bool)
            candidate_tokens = self.process_scores(input_ids, 0)
            full_mask[:, candidate_tokens] = False
            scores[full_mask] = -torch.inf

        return scores

    def is_bos_token_in_input(self, input_ids):
        """Check if the input contains a BOS token. Checking the first sequence is enough."""
        return (input_ids[0, 0] == self.bos_token_id).item()

    def is_next_token_entity(self, input_ids):
        current_len = input_ids.shape[-1]
        has_bos_token = self.is_bos_token_in_input(input_ids)

        # bos_token determines if the current length is even or odd
        return current_len % 2 == has_bos_token

    def process_scores(self, input_ids, idx):
        """Process each score based on input length and update mask to allow only entities or only relations."""
        current_len = input_ids.shape[-1]
        has_bos_token = self.is_bos_token_in_input(input_ids)

        if current_len == self.max_sequence_length - 1 - has_bos_token:
            current_uid = input_ids[idx, int(has_bos_token)].item()
            candidate_tokens = self.pos_candidates_cache.get(current_uid)
            if candidate_tokens is None:
                candidate_tokens = np.arange(len(self.tokenizer))

                user_used_ids = self.tokenized_used_ids[current_uid]
                candidate_tokens = np.setdiff1d(candidate_tokens, list(user_used_ids), assume_unique=True)
                self.pos_candidates_cache[current_uid] = candidate_tokens
        elif self.is_next_token_entity(input_ids):
            candidate_tokens = self.entity_token_ids
        else:
            candidate_tokens = self.relation_token_ids

        return candidate_tokens
