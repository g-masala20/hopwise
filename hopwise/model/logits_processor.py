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

    RECOMMENDATION_TASK = "recommendation"
    LINK_PREDICTION_TASK = "link_prediction"

    def __init__(
        self,
        tokenized_kg,
        tokenized_ignored_ids,
        max_sequence_length,
        tokenizer,
        num_return_sequences,
        train_data,  # aggiunta mia 
        mask_cache_size=3 * 10**4,
        pos_candidates_cache_size=1 * 10**5,
        task="recommendation",
        **kwargs,
    ):
        super().__init__(
        tokenized_kg,
        tokenized_ignored_ids,
        max_sequence_length,
        tokenizer,
        num_return_sequences,
        mask_cache_size,
        pos_candidates_cache_size,
        task,
        **kwargs)
        self.train_data = train_data  # Salva train_data come attributo dell'istanza

    def __call__(self, input_ids, scores):

        #problemi del codice:
        # sembra che la fullmask sia sempre piena, quindi il codice non va mai avanti
        # va fatto il file di configurazione (restrictions_ml-100k.yaml) 
        # e ho modificato direttamente il dataset ml-100k (probabilmente male)
        
        
        # finire le modifiche all'applicazione delle restrizioni post-aggiornamento da casuale a file-definied
        
        train_data = self.train_data
        entity_mapping = train_data.dataset.field2token_id['entity_id'] #todo passa solo questo


        #breakpoint()
        current_len = input_ids.shape[-1]
        has_bos_token = self.is_bos_token_in_input(input_ids)

        if has_bos_token and current_len == self.max_sequence_length - 1:
            self.mask_non_eos_tokens(scores)
        else:
            unique_input_ids = input_ids
            if self.task == self.RECOMMENDATION_TASK and current_len < self.max_sequence_length - 1 - has_bos_token:
                user_idx = 1
                
                _, input_ids_indices, input_ids_inv = np.unique(
                    input_ids.cpu().numpy()[:, [user_idx]], axis=0, return_index=True, return_inverse=True
                )
                unique_input_ids = input_ids[input_ids_indices]

            # ---

            #all_entity_keys = list(entity_mapping.keys())

            # Estrai i constraint dal dataset (con fallback se non esiste)
            try:
                constraints_tokens = train_data.dataset.field2id_token['constraints']
                #print(f"DEBUG: constraints found! Length: {len(constraints_tokens)}")

                # Solo se i constraints esistono, procedi con il sistema di restrizioni
                hard_restriction_keys_per_user = []
                soft_restriction_keys_per_user = []
                
                for idx in range(unique_input_ids.shape[0]):
                    user_position = 1 if has_bos_token else 0
                    user_idx_in_batch = unique_input_ids[idx, user_position].item()
                    
                    if user_idx_in_batch < len(constraints_tokens):
                        user_constraints = constraints_tokens[user_idx_in_batch].split(',') if constraints_tokens[user_idx_in_batch] else []
                        
                        # Primi 2 constraint per hard restrictions
                        hard_constraints = user_constraints[:2] if len(user_constraints) >= 2 else []
                        # Terzo e quarto constraint per soft restrictions  
                        soft_constraints = user_constraints[2:4] if len(user_constraints) >= 4 else []
                        
                        hard_restriction_keys_per_user.append(hard_constraints)
                        soft_restriction_keys_per_user.append(soft_constraints)
                    else:
                        # Fallback per utenti senza constraints
                        hard_restriction_keys_per_user.append([])
                        soft_restriction_keys_per_user.append([])
                        
                # maschera completa, contentente le maschere l'applicazione di tute le restrizioni
                full_mask = np.zeros((unique_input_ids.shape[0], len(self.tokenizer)), dtype=bool)
                
            except KeyError:
                # Se 'constraints' non esiste, salta tutto il sistema di constraints
                constraints_tokens = None
                hard_restriction_keys_per_user = [[] for _ in range(unique_input_ids.shape[0])]
                soft_restriction_keys_per_user = [[] for _ in range(unique_input_ids.shape[0])]
                full_mask = np.zeros((unique_input_ids.shape[0], len(self.tokenizer)), dtype=bool) # Inizializza maschera vuota che non blocca nulla

            # ---

            for idx in range(unique_input_ids.shape[0]):
                if self.task == self.RECOMMENDATION_TASK:
                    key, candidate_tokens = self.process_scores_rec(unique_input_ids, idx)
                elif self.task == self.LINK_PREDICTION_TASK:
                    key, candidate_tokens = self.process_scores_lp(unique_input_ids, idx)
                
                # HARD MASKING:

                # Inizializza la maschera base una sola volta
                banned_mask = self.get_banned_mask(key, candidate_tokens)
                full_mask[idx] = banned_mask.copy()
                
                # DEBUG: Controlla se già mascherato dalla logica base
                if np.all(full_mask[idx]):
                    print(f"DEBUG: Base mask already blocks all tokens for idx={idx} (BEFORE constraints)")
                
                # Applica hard restrictions solo se i constraints esistono
                if constraints_tokens is not None:
                    hard_restriction_mask = np.zeros(len(self.tokenizer), dtype=bool)
                    break
                    
                    # Accumula tutte le restrizioni hard per questo utente
                    for x_val in hard_restriction_keys_per_user[idx]:
                        constraint_mask = self.gen_banmask_from_key(x_val, train_data)
                        hard_restriction_mask = np.logical_or(hard_restriction_mask, constraint_mask)

                        if np.all(full_mask[idx]): 
                            print(f"DEBUG: All tokens masked for idx={idx} (AFTER constraints) (dentro il ciclo)")
                            raise RuntimeError("All tokens are masked for all input rows in full_mask.(dentro il ciclo)")

                    
                    # Combina maschera base con restrizioni hard
                    full_mask[idx] = np.logical_or(banned_mask, hard_restriction_mask)

                if np.all(banned_mask): #np.all(full_mask[idx]): 
                    print(f"DEBUG: All tokens masked for idx={idx} (AFTER constraints)")
                    raise RuntimeError("All tokens are masked for all input rows in full_mask.(BANANAAAAAA)")

                #--- 

                #SOFT MASKING:
                # Applica soft masking solo se i constraints esistono
                if constraints_tokens is not None:
                    # sintesi: vengono ordinati i valori in modo decrescente in base al numero di token connessi,
                    # viene applicata la maschera per volta, e si applica la successiva solo se il kg è valido,
                    # altrimenti si interrompe il processo 
                    
                    # Solo ordinare se ci sono effettivamente constraint soft
                    if soft_restriction_keys_per_user[idx]:
                        soft_restriction_keys_per_user[idx] = sorted(
                            soft_restriction_keys_per_user[idx],
                            key=lambda k: len(self.tokenized_kg[entity_mapping[k]]) if k in entity_mapping and entity_mapping[k] in self.tokenized_kg else 0,
                            reverse=True,
                        )

                    #soft_restriction_mask[idx] = self.get_banned_mask(key, candidate_tokens)

                    for y_val in soft_restriction_keys_per_user[idx]:
                        # Verifica che y_val esista nell'entity_mapping
                        if y_val in entity_mapping:
                            soft_mask = np.logical_or(full_mask[idx], self.gen_banmask_from_key(y_val, train_data))
                            if np.all(soft_mask): 
                                break
                            else:
                                full_mask[idx] = soft_mask

            #---

            if self.task == self.RECOMMENDATION_TASK and current_len < self.max_sequence_length - 1 - has_bos_token:
                scores[full_mask[input_ids_inv]] = -math.inf
            else:
                scores[full_mask] = -math.inf

        return scores
    
    def generate_and_save_constraints(self, user_file_path, n_constraints=6, output_file_path=None):
        """
        Aggiorna il file utente aggiungendo la colonna 'constraints',
        per ogni utente genera n_constraints restrizioni casuali da entity_mapping.
        Se output_file_path è None, sovrascrive user_file_path, altrimenti salva su output_file_path.
        """
        import random
        
        entity_mapping = self.train_data.dataset.field2token_id['entity_id']
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


    def __call__WRONG(self, input_ids, scores):

        #problemi del codice:
        # tanti e sono tutti del programmatore :thumbs_up:

        #hard_restriction_keys = ["m.0v1k9hr", "m.0zbg8vq", "m.0zdxbv0", "m.0znj_66"] 
        #soft_restriction_keys = ["m.0zcbl", "m.0zhv", "m.0z4s", "m.0yzbg"]
        #preferences_keys = ["m.0v1k9hr", "m.0zbg8vq", "m.0zdxbv0", "m.0znj_66"]
        
        train_data = self.train_data
        entity_mapping = train_data.dataset.field2token_id['entity_id'] #todo passa solo questo

        #breakpoint()
        current_len = input_ids.shape[-1]
        has_bos_token = self.is_bos_token_in_input(input_ids)

        if has_bos_token and current_len == self.max_sequence_length - 1:
            self.mask_non_eos_tokens(scores)
        else:
            unique_input_ids = input_ids
            if self.task == self.RECOMMENDATION_TASK and current_len < self.max_sequence_length - 1 - has_bos_token:
                user_idx = 1
                #breakpoint()
                _, input_ids_indices, input_ids_inv = np.unique(
                    input_ids.cpu().numpy()[:, [user_idx]], axis=0, return_index=True, return_inverse=True
                )
                unique_input_ids = input_ids[input_ids_indices]

            # ---
            # estrazione delle chiavi per le restrizioni per utente 

            all_entity_keys = list(entity_mapping.keys())

            hard_restriction_keys_per_user = [
                random.sample(all_entity_keys, 5) for _ in range(unique_input_ids.shape[0])
            ]

            soft_restriction_keys_per_user = [
                random.sample(all_entity_keys, 5) for _ in range(unique_input_ids.shape[0])
            ]

            preferences_keys_per_user = [
                random.sample(all_entity_keys, 5) for _ in range(unique_input_ids.shape[0])
            ]
           
            with open("hr_vals.txt", "w") as f:
                for sotto_lista in hard_restriction_keys_per_user:
                    f.write(" ".join(map(str, sotto_lista)) + "\n")

            # ---
            breakpoint()
            full_mask = [ np.zeros((unique_input_ids.shape[0], len(self.tokenizer)), dtype=bool) for _ in range(unique_input_ids.shape[0]) ]

            for idx in range(unique_input_ids.shape[0]):

                hard_restriction_mask = [ np.zeros(len(self.tokenizer), dtype=bool) for _ in range(unique_input_ids.shape[0]) ]
                
                if self.task == self.RECOMMENDATION_TASK:
                    key, candidate_tokens = self.process_scores_rec(unique_input_ids, idx)
                elif self.task == self.LINK_PREDICTION_TASK:
                    key, candidate_tokens = self.process_scores_lp(unique_input_ids, idx)
                
                # HARD MASKING:
                # Per ogni utente, usa la lista di hard_restriction_keys_per_user[idx]
                
                # Inizializza la maschera base una sola volta
                banned_mask = self.get_banned_mask(key, candidate_tokens)
                hard_restriction_mask = np.zeros(len(self.tokenizer), dtype=bool)
                
                # Accumula tutte le restrizioni hard per questo utente
                for x_val in hard_restriction_keys_per_user[idx]:
                    if x_val in candidate_tokens:
                        constraint_mask = self.gen_banmask_from_key(x_val, train_data)
                        hard_restriction_mask = np.logical_or(hard_restriction_mask, constraint_mask)
                
                # Combina maschera base con restrizioni hard
                full_mask[idx] = np.logical_or(banned_mask, hard_restriction_mask)

                if np.all(full_mask):
                    raise RuntimeError("All tokens are masked for all input rows in full_mask.")

                #--- 

                #SOFT MASKING:
                # sintesi: vengono ordinati i valori in modo decrescente in base al numero di token connessi,
                # viene applicata la maschera per volta, e si applica la successiva solo se il kg è valido,
                # altrimenti si interrompe il processo 
                for i, sr_keys_user in enumerate(soft_restriction_keys_per_user[idx]):
                    soft_restriction_keys_per_user[idx] = sorted(
                        sr_keys_user,
                        key=lambda k: len(self.tokenized_kg[entity_mapping[k]]) if entity_mapping[k] in self.tokenized_kg else 0,
                        reverse=True,
                    )

                soft_restriction_mask = [ self.get_banned_mask(key, candidate_tokens) for _ in range(unique_input_ids.shape[0]) ]
                tmp_mask = full_mask.copy() 
                #breakpoint()
                for i,sr_keys_user in enumerate(soft_restriction_keys_per_user[idx]):
                    for y_val in sr_keys_user:
                        if y_val in candidate_tokens:
                            soft_restriction_mask[i] = np.logical_or(soft_restriction_mask[i], self.gen_banmask_from_key(y_val, train_data))
                            tmp_mask[i][idx] = np.logical_or(tmp_mask[idx], soft_restriction_mask)
                            
                            if np.all(tmp_mask): 
                                break
                            else:
                                full_mask[i][idx] = tmp_mask[idx]

            #---

            # if self.task == self.RECOMMENDATION_TASK and current_len < self.max_sequence_length - 1 - has_bos_token:
            #     scores[full_mask[input_ids_inv]] = -math.inf
            # else:
            #     scores[full_mask] = -math.inf

        return scores

    def extract_connected_entities(self, token_id):
        """
        Estrae tutte le entità connesse al token_id, sotto forma di lista completamente appiattita.
        """
        connected_entities = set()
        for entity_set in self.tokenized_kg[token_id].values():
            connected_entities.update(entity_set if isinstance(entity_set, (list, set)) else [entity_set])
        connected_entities = list(set(connected_entities))#.sorted()  # l'ordine è temporaneo, serve solo a render più leggibile l'output
        
        return connected_entities
    
    def gen_banmask_from_key(self, key, train_data, ban_connected_entities=False):

        mask = np.zeros(len(self.tokenizer), dtype=bool)

        id_interno = train_data.dataset.field2token_id['entity_id'][key] #todo: passo troppe cose, si può migliorare
        if id_interno < train_data.dataset.item_num:
            token = PathLanguageModelingTokenType.ITEM.value + str(id_interno)
        else:
            token = PathLanguageModelingTokenType.ENTITY.value + str(id_interno)

        token_id = self.tokenizer.convert_tokens_to_ids(token)
        mask[token_id] = True
        print(f"DEBUG GEN_BANMASK: Key '{key}' -> banned main token (ID: {token_id})")

        # Ban di tutte le entità connesse per ogni key (solo se abilitato)
        if ban_connected_entities and token_id in self.tokenized_kg:
            connected_entities = self.extract_connected_entities(token_id)
            print(f"DEBUG GEN_BANMASK: Key '{key}' also bans {len(connected_entities)} connected entities")
            mask[connected_entities] = True  # ban di tutte le entità connesse
        elif not ban_connected_entities:
            print(f"DEBUG GEN_BANMASK: Key '{key}' -> connected entities ban DISABLED")

        return mask
    
    def __call__SINGLEUSER(self, input_ids, scores):

        #problemi del codice:
        # tanti e sono tutti del programmatore :thumbs_up:

        hard_restriction_keys = ["m.0v1k9hr", "m.0zbg8vq", "m.0zdxbv0", "m.0znj_66"] 
        soft_restriction_keys = ["m.0zcbl", "m.0zhv", "m.0z4s", "m.0yzbg"]
        preferences_keys = ["m.0v1k9hr", "m.0zbg8vq", "m.0zdxbv0", "m.0znj_66"]
        
        train_data = self.train_data
        entity_mapping = train_data.dataset.field2token_id['entity_id'] #todo passa solo questo

        #breakpoint()
        current_len = input_ids.shape[-1]
        has_bos_token = self.is_bos_token_in_input(input_ids)

        if has_bos_token and current_len == self.max_sequence_length - 1:
            self.mask_non_eos_tokens(scores)
        else:
            unique_input_ids = input_ids
            if self.task == self.RECOMMENDATION_TASK and current_len < self.max_sequence_length - 1 - has_bos_token:
                user_idx = has_bos_token
                #breakpoint()
                _, input_ids_indices, input_ids_inv = np.unique(
                    input_ids.cpu().numpy()[:, [user_idx]], axis=0, return_index=True, return_inverse=True
                )
                unique_input_ids = input_ids[input_ids_indices]



            full_mask = np.zeros((unique_input_ids.shape[0], len(self.tokenizer)), dtype=bool) 

            hard_restriction_mask = np.zeros(len(self.tokenizer), dtype=bool)
            
            for idx in range(unique_input_ids.shape[0]):
                if self.task == self.RECOMMENDATION_TASK:
                    key, candidate_tokens = self.process_scores_rec(unique_input_ids, idx)
                elif self.task == self.LINK_PREDICTION_TASK:
                    key, candidate_tokens = self.process_scores_lp(unique_input_ids, idx)

                # HARD MASKING:
                # sintesi: 
                # vengono applicate tutte le maschere, se il kg è valido allora andiamo avanti, 
                # altrimenti il processo non può continuare e viene lanciato un errore
                for x_val in hard_restriction_keys:
                    if x_val in candidate_tokens:
                        hard_restriction_mask = np.logical_or(hard_restriction_mask, self.gen_banmask_from_key(x_val, train_data))
                                
                banned_mask = self.get_banned_mask(key, candidate_tokens)
                full_mask[idx] = np.logical_or(banned_mask, hard_restriction_mask) 

                if np.all(full_mask):
                    raise RuntimeError("All tokens are masked for all input rows in full_mask.")

                #--- 

                #SOFT MASKING:
                # sintesi: vengono ordinati i valori in modo decrescente in base al numero di token connessi,
                # viene applicata la maschera per volta, e si applica la successiva solo se il kg è valido,
                # altrimenti si interrompe il processo 
                soft_restriction_keys = sorted(
                    soft_restriction_keys,
                    key=lambda k: len(self.tokenized_kg[entity_mapping[k]]) if entity_mapping[k] in self.tokenized_kg else 0,
                    reverse=True,
                )

                soft_restriction_mask = self.get_banned_mask(key, candidate_tokens)
                tmp_mask = full_mask.copy() 
                #breakpoint()
                for y_val in soft_restriction_keys:
                    if y_val in candidate_tokens:
                        soft_restriction_mask = np.logical_or(soft_restriction_mask, self.gen_banmask_from_key(y_val, train_data))
                        tmp_mask[idx] = np.logical_or(tmp_mask[idx], soft_restriction_mask)
                        
                        if np.all(tmp_mask): 
                            break
                        else:
                            full_mask[idx] = tmp_mask[idx]

            #---

            if self.task == self.RECOMMENDATION_TASK and current_len < self.max_sequence_length - 1 - has_bos_token:
                scores[full_mask[input_ids_inv]] = -math.inf
            else:
                scores[full_mask] = -math.inf

        return scores
                                
    def __call__MULTIPLO(self, input_ids, scores):

        x = ["m.0v1k9hr", "m.0zbg8vq", "m.0zdxbv0", "m.0znj_66"] # ora x può essere una lista di valori da escludere
        train_data = self.train_data
        entity_mapping = train_data.dataset.field2token_id['entity_id']

        #breakpoint()
        current_len = input_ids.shape[-1]
        has_bos_token = self.is_bos_token_in_input(input_ids)

        if has_bos_token and current_len == self.max_sequence_length - 1:
            self.mask_non_eos_tokens(scores)
        else:
            unique_input_ids = input_ids
            if self.task == self.RECOMMENDATION_TASK and current_len < self.max_sequence_length - 1 - has_bos_token:
                last_n_tokens = 2 if self.is_next_token_entity(input_ids) else 1
                _, input_ids_indices, input_ids_inv = np.unique(
                    input_ids.cpu().numpy()[:, -last_n_tokens:], axis=0, return_index=True, return_inverse=True
                )
                unique_input_ids = input_ids[input_ids_indices]

            full_mask = np.zeros((unique_input_ids.shape[0], len(self.tokenizer)), dtype=bool)

            for idx in range(unique_input_ids.shape[0]):

                hard_restriction_mask = np.zeros(len(self.tokenizer), dtype=bool)

                if self.task == self.RECOMMENDATION_TASK:
                    key, candidate_tokens = self.process_scores_rec(unique_input_ids, idx)

                    for x_val in x:

                        if x_val in candidate_tokens:
                            id_interno_x = entity_mapping[x_val]
                            if id_interno_x < train_data.dataset.item_num:
                                token = PathLanguageModelingTokenType.ITEM.value + str(id_interno_x)
                            else:
                                token = PathLanguageModelingTokenType.ENTITY.value + str(id_interno_x)

                            token_id = self.tokenizer.convert_tokens_to_ids(token)
                            hard_restriction_mask[token_id] = True

                            # Ban di tutte le entità connesse ad x_val
                            if id_interno_x in self.tokenized_kg:
                                connected_entities = self.tokenized_kg[id_interno_x].keys()
                                for connected_entity in connected_entities:
                                    connected_token = PathLanguageModelingTokenType.ENTITY.value + str(connected_entity)
                                    connected_token_id = self.tokenizer.convert_tokens_to_ids(connected_token)
                                    hard_restriction_mask[connected_token_id] = True

                elif self.task == self.LINK_PREDICTION_TASK:
                    key, candidate_tokens = self.process_scores_lp(unique_input_ids, idx)

                banned_mask = self.get_banned_mask(key, candidate_tokens)
                full_mask[idx] = np.logical_or(banned_mask, hard_restriction_mask) # si può migliorare?

                if np.all(full_mask): # probabilmente molto lento
                    raise RuntimeError("All tokens are masked for all input rows in full_mask.")

            if self.task == self.RECOMMENDATION_TASK and current_len < self.max_sequence_length - 1 - has_bos_token:
                scores[full_mask[input_ids_inv]] = -math.inf
            else:
                scores[full_mask] = -math.inf

        return scores
    
    def __call__SINGOLO(self, input_ids, scores):

        #---
        x = "m.0v1k9hr" # valore che vogliamo non avere tra i candidati, che verrebbe preso con API
        train_data = self.train_data
        entity_mapping = train_data.dataset.field2token_id['entity_id']
        #z = [] # variabile per il testing, da eliminare dopo
        #---
        #breakpoint()
        current_len = input_ids.shape[-1] # current_len: lunghezza delle connessioni tra elementi tokenizzati 
        has_bos_token = self.is_bos_token_in_input(input_ids)

        if has_bos_token and current_len == self.max_sequence_length - 1:
            self.mask_non_eos_tokens(scores) # scores: valori del logits [batch_size, n_tokens]
        else:
            unique_input_ids = input_ids
            if self.task == self.RECOMMENDATION_TASK and current_len < self.max_sequence_length - 1 - has_bos_token: # se la lung non è ancora quella max
                last_n_tokens = 2 if self.is_next_token_entity(input_ids) else 1
                _, input_ids_indices, input_ids_inv = np.unique(
                    input_ids.cpu().numpy()[:, -last_n_tokens:], axis=0, return_index=True, return_inverse=True
                )
                unique_input_ids = input_ids[input_ids_indices] 

            full_mask = np.zeros((unique_input_ids.shape[0], len(self.tokenizer)), dtype=bool) # maschera per ogni elemento, in modo da abilitare solo le connessioni utili
            
            
            for idx in range(unique_input_ids.shape[0]):

                #---
                hard_restriction_mask = np.zeros(len(self.tokenizer), dtype=bool) 
                #---

                if self.task == self.RECOMMENDATION_TASK:
                    key, candidate_tokens = self.process_scores_rec(unique_input_ids, idx)

                    #---
                    # aggiorno ma maschera ber bannare i token hard-No
                    if x in candidate_tokens:
                        id_interno_x = entity_mapping[x]
                        if id_interno_x < train_data.dataset.item_num:
                            token = PathLanguageModelingTokenType.ITEM.value + str(id_interno_x) # verifichiamo se dobbiamo escludere un item
                        else:
                            token = PathLanguageModelingTokenType.ENTITY.value + str(id_interno_x) # altrimenti dobbiamo escludere un'entità
                            
                        token_id = self.tokenizer.convert_tokens_to_ids(token)
                        hard_restriction_mask[token_id] = True
                        #breakpoint()
                        # Ban di tutte le entità connesse ad x
                        if id_interno_x in self.tokenized_kg:
                            connected_entities = self.tokenized_kg[id_interno_x].keys()
                            #z = connected_entities # testing, da eliminare dopo
                            for connected_entity in connected_entities:
                                connected_token = PathLanguageModelingTokenType.ENTITY.value + str(connected_entity)
                                connected_token_id = self.tokenizer.convert_tokens_to_ids(connected_token)
                                hard_restriction_mask[connected_token_id] = True
                    #---

                elif self.task == self.LINK_PREDICTION_TASK:
                    key, candidate_tokens = self.process_scores_lp(unique_input_ids, idx)

                banned_mask = self.get_banned_mask(key, candidate_tokens) # assegna a banned_mask gli id dei token che non possono essere scelti
                full_mask[idx] = banned_mask

                #---
                full_mask[idx] = np.logical_or(banned_mask, hard_restriction_mask)
                #---

            if self.task == self.RECOMMENDATION_TASK and current_len < self.max_sequence_length - 1 - has_bos_token:
                scores[full_mask[input_ids_inv]] = -math.inf
            else:
                scores[full_mask] = -math.inf

        #--- testing testing testing 
        # z_token_id = []
        # for connected_entity in z: #(z = connected_entities)
        #    z_token = PathLanguageModelingTokenType.ENTITY.value + str(connected_entity)
        #    z_token_id.append(self.tokenizer.convert_tokens_to_ids(connected_token))

        # if z_token_id and not torch.isinf(scores[:, z_token_id]).all():
        #    raise ValueError("Some token scores in the list z are not -inf as expected.")
        #--- testing testing testing 

        # se la maschera è tutte a true, non abbiamo più un kg valido
        if np.all(full_mask):
            raise RuntimeError("All tokens are masked for all input rows in full_mask.")

        return scores



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
