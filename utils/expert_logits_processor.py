from functools import partial
from typing import List
import torch
import numpy as np
from transformers import LogitsProcessor
import time
from queue import Queue
from threading import Thread
import copy

class ExternalExpertsLogitsProcessor(LogitsProcessor):
    
    def __init__(self, lm_tokenizer, reward_model, topk=20, 
                 method="linear", beta=30, inverse=False, generation_log=None, data_container=None):
        self._lm_tokenizer = lm_tokenizer
        self._reward_model = reward_model.to('cuda')
        self._reward_model.eval()
        self._topk = topk
        self._method = method
        self._beta = beta
        self._inverse = inverse
        self._generation_log = generation_log

        self._data_contrainer = data_container
        self._step = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        logits = self._reward_model(input_ids, labels=None, use_cache=True, past_key_values=self._data_contrainer)
        logits = logits[:, -1, :]

        _, topk_ids = torch.topk(scores, self._topk, dim=-1)    

        if self._generation_log is not None:
            for batch_id in range(input_ids.shape[0]):
                log = []
                for it, idx in enumerate(topk_ids[batch_id]):
                    rs = logits[batch_id, it]
                    score = scores[batch_id, idx]
                    prefix = self._lm_tokenizer.decode(input_ids[batch_id].tolist())
                    log.append({'prefix': prefix, 'rs': rs.item(), 'score': score.item(), 'idx': idx.item()})
                self._generation_log.extend(log)

        
        selected_logits_orig = scores.gather(1, topk_ids)
        selected_logits_rm = logits.gather(1, topk_ids)
        new_scores = torch.full_like(scores, -float("Inf"))
        scores = self.apply_function(selected_logits_orig, selected_logits_rm)
        new_scores.scatter_(1, topk_ids, scores)
        return new_scores
    
    def apply_function(self, original_score, reward_score):
        if self._inverse:
            reward_score = 1-reward_score
        if self._method == "linear":
            return original_score + (reward_score*self._beta).to(original_score.dtype)
        else:
            raise ValueError(f"method {self._method} not supported")
    