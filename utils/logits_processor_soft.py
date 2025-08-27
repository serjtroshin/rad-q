from functools import partial
import os
import random
from typing import List
import torch
import numpy as np
from transformers import LogitsProcessor, TopPLogitsWarper, TopKLogitsWarper
import time
from queue import Queue
from threading import Thread
import copy


def parse_decoding_kwargs(decoding_kwargs):
    if decoding_kwargs.get("topk", None) is not None:
        topk = decoding_kwargs["topk"]
        return TopKLogitsWarper(top_k=topk)
    elif decoding_kwargs.get("top_p", None) is not None:
        top_p = decoding_kwargs["top_p"]
        return TopPLogitsWarper(top_p=top_p)
    else:
        return None
    

class RewardAugmentedLogitsSoftProcessor(LogitsProcessor):

    def __init__(self, lm_tokenizer, reward_model, topk=-1, 
                 method="linear", beta=30, inverse=False, crop_probs=True, stop_function="none", stop_function_coef=0.2, data_container=None, generation_log=None,
                 prepend_bos=True):
        self._lm_tokenizer = lm_tokenizer
        self._rm_tokenizer = lm_tokenizer   # need to be the same!
        self._reward_model = reward_model.to('cuda')
        self._reward_model.eval()
        self._topk = topk
        self._method = method
        self._beta = beta
        self._inverse = inverse

        self._step = 0

        # ablation args
        self._crop_probs = crop_probs
        self._stop_function = stop_function
        self._stop_function_coef = stop_function_coef

        self._generation_log = generation_log

        self.past_key_values = data_container
        self.bos_token = lm_tokenizer.bos_token_id

        self.prepend_bos = prepend_bos  # prepend bos token if cache is empty

    def extract_past_key_values(self, input_ids):
        with torch.inference_mode():
            _, _, past_key_values = self._reward_model(input_ids, labels=None, use_cache=True)
            return past_key_values

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        input_ids: (batch, seq_len) token_ids
        scores: (batch, vocab_size) tensor with -inf for tokens out of support of a decoding algorithm
        """
        if self._topk == -1:
            # print average number of non-inf tokens:
            if random.random() < 0.01:
                print("average number of non-inf tokens", torch.sum(torch.isfinite(scores)).item() / scores.numel() * scores.shape[1])

            vocab_size = scores.shape[1]
            _, topk_ids = torch.topk(scores, vocab_size, dim=-1)   # use top 1024 tokens
            # take all ids
            # topk_ids = torch.arange(scores.shape[1], device=scores.device).unsqueeze(0).expand(input_ids.shape[0], -1)
        else:
            _, topk_ids = torch.topk(scores, self._topk, dim=-1)                                    # (batch, topk,)
            # for top_k we can pass efficiently only relevant ids, for other decoding algorithms we need to pass all ids

        if self.prepend_bos:
            input_ids = torch.cat((torch.tensor([[self.bos_token]]*input_ids.shape[0], device=input_ids.device), input_ids, ), dim=1)

        if self._topk == self._lm_tokenizer.vocab_size:
            # debug model
            next_tokens_range = torch.arange(self._lm_tokenizer.vocab_size, device='cuda').unsqueeze(0)
            reward_scores, self.past_key_values = self.get_reward(input_ids, self.past_key_values, index=next_tokens_range)
            # convert to list
            self._generation_log.append({
                'input_ids': self._lm_tokenizer.decode(input_ids[0]),
                'all_rewards': reward_scores.cpu().squeeze().tolist(),
                'all_lm_scores': scores.cpu().squeeze().tolist(),
            })
            return scores

        reward_scores, self.past_key_values = self.get_reward(input_ids, self.past_key_values, index=topk_ids)

        reward_scores = self.apply_function(scores, reward_scores, index=topk_ids)
        # apply top-k filtering from base lm using topk_ids
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(1, topk_ids, True)
        reward_scores[~mask] = -float("Inf")
        return reward_scores
    
    def get_reward(self, input_ids, past_key_values=None, index=None):
        # calculated reward scores for all input_ids
        with torch.inference_mode():
            if past_key_values is not None:
                input_ids = input_ids[:, -1:]

            index = index.unsqueeze(1).expand(index.shape[0], input_ids.shape[1], -1)

            rewards, past_key_values = self._reward_model(input_ids=input_ids, labels=None, use_cache=True, past_key_values=past_key_values, querry=index)
            rewards = rewards[:, -1, 0, :]  # (batch, k)
            assert past_key_values is not None
            return rewards, past_key_values
    
    def apply_function(self, scores, reward_scores, index=None):
        if self._crop_probs:
            reward_scores = torch.clamp(reward_scores, min=0, max=1)

        if self._inverse:
            reward_scores = 1-reward_scores
        if self._method == "linear":
            if index is None:
                return scores + reward_scores*self._beta
            else:
                # add by index to scores
                scores.scatter_add_(1, index, reward_scores*self._beta)
                return scores
        else:
            raise ValueError(f"method {self._method} not supported")
    

class RewardAugmentedLogitsSoftProcessorOrig(LogitsProcessor):

    def __init__(self, lm_tokenizer, reward_model, topk=20, 
                 method="linear", beta=30, inverse=False, crop_probs=True, stop_function="none", stop_function_coef=0.2, data_container=None, generation_log=None,
                 prepend_bos=True):
        self._lm_tokenizer = lm_tokenizer
        self._rm_tokenizer = lm_tokenizer   # need to be the same!
        self._reward_model = reward_model.to('cuda')
        self._reward_model.eval()
        self._topk = topk
        self._method = method
        self._beta = beta
        self._inverse = inverse

        self._step = 0

        # ablation args
        self._crop_probs = crop_probs
        self._stop_function = stop_function
        self._stop_function_coef = stop_function_coef

        self._generation_log = generation_log

        self.past_key_values = data_container
        self.bos_token = lm_tokenizer.bos_token_id

        self.prepend_bos = prepend_bos  # prepend bos token if cache is empty

    def extract_past_key_values(self, input_ids):
        with torch.inference_mode():
            _, _, past_key_values = self._reward_model(input_ids, labels=None, use_cache=True)
            return past_key_values

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        input_ids: (batch, seq_len) token_ids
        scores: (batch, vocab_size) tensor with -inf for tokens out of support of a decoding algorithm
        """
        _, topk_ids = torch.topk(scores, self._topk, dim=-1)                                    # (batch, topk,)

        if self.prepend_bos:
            input_ids = torch.cat((torch.tensor([[self.bos_token]]*input_ids.shape[0], device=input_ids.device), input_ids, ), dim=1)


        reward_scores, self.past_key_values = self.get_reward(input_ids, self.past_key_values, index=topk_ids)

        if self._generation_log is not None:
            for batch_id in range(input_ids.shape[0]):
                log = []
                for it, idx in enumerate(topk_ids[batch_id]):
                    rs = reward_scores[batch_id, it]
                    score = scores[batch_id, idx]
                    prefix = self._lm_tokenizer.decode(input_ids[batch_id].tolist())
                    log.append({'prefix': prefix, 'rs': rs.item(), 'score': score.item(), 'idx': idx.item()})
                self._generation_log.extend(log)

        reward_scores = self.apply_function(scores, reward_scores, index=topk_ids)
        # apply top-k filtering from base lm using topk_ids
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(1, topk_ids, True)
        reward_scores[~mask] = -float("Inf")

        return reward_scores
    
    def get_reward(self, input_ids, past_key_values=None, index=None):
        # calculated reward scores for all input_ids
        with torch.inference_mode():
            if past_key_values is not None:
                input_ids = input_ids[:, -1:]

            index = index.unsqueeze(1).expand(index.shape[0], input_ids.shape[1], -1)

            rewards, past_key_values = self._reward_model(input_ids=input_ids, labels=None, use_cache=True, past_key_values=past_key_values, querry=index)
            rewards = rewards[:, -1, 0, :]  # (batch, k)
            assert past_key_values is not None
            return rewards, past_key_values
    
    def apply_function(self, scores, reward_scores, index=None):
        if self._stop_function == "none":
            pass
        elif self._stop_function == "crop_uncertain":
            reward_scores = torch.where(reward_scores<self._stop_function_coef, torch.zeros_like(reward_scores), reward_scores)
            raise NotImplementedError("crop_uncertain is not supported")

        if self._crop_probs:
            reward_scores = torch.clamp(reward_scores, min=0, max=1)

        if self._inverse:
            reward_scores = 1-reward_scores
        if self._method == "linear":
            if index is None:
                return scores + reward_scores*self._beta
            else:
                # add by index to scores
                scores.scatter_add_(1, index, reward_scores*self._beta)
                return scores
        else:
            raise ValueError(f"method {self._method} not supported")

class RewardAugmentedLogitsSoftProcessorNoPkv(LogitsProcessor):
    
    def __init__(self, lm_tokenizer, reward_model, topk=20, 
                 method="linear", beta=30, inverse=False, crop_probs=True, stop_function="none", stop_function_coef=0.2, generation_log=None):
        self._lm_tokenizer = lm_tokenizer
        # self._rm_tokenizer = rm_tokenizer
        self._reward_model = reward_model.to('cuda')
        self._reward_model.eval()
        self._topk = topk
        self._method = method
        self._beta = beta
        self._inverse = inverse

        # ablation args
        self._crop_probs = crop_probs
        self._stop_function = stop_function
        self._stop_function_coef = stop_function_coef

        self._generation_log = generation_log

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        _, topk_ids = torch.topk(scores, self._topk, dim=-1)     
        
        # return reward scores
        reward_scores = self.get_reward(input_ids)

        if self._generation_log is not None:
            for batch_id in range(input_ids.shape[0]):
                log = []
                for idx in topk_ids[batch_id]:
                    rs = reward_scores[batch_id, idx]
                    score = scores[batch_id, idx]
                    prefix = self._lm_tokenizer.decode(input_ids[batch_id].tolist())
                    log.append({'prefix': prefix, 'rs': rs.item(), 'score': score.item(), 'idx': idx.item()})
                self._generation_log.extend(log)


        reward_scores = self.apply_function(scores, reward_scores)
        # apply top-k filtering from base lm using topk_ids
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(1, topk_ids, True)
        reward_scores[~mask] = -float("Inf")

        return reward_scores
    
    def get_reward(self, input_ids):
        # calculated reward scores for all input_ids
        with torch.inference_mode():
            rewards = self._reward_model(input_ids=input_ids, labels=None)[:, -1, 0, :]  # (batch, vocab_size)
            return rewards
    
    def apply_function(self, scores, reward_scores):
        if self._stop_function == "none":
            pass
        elif self._stop_function == "crop_uncertain":
            reward_scores = torch.where(reward_scores<self._stop_function_coef, torch.zeros_like(reward_scores), reward_scores)
            raise NotImplementedError("crop_uncertain is not supported")
        
        if self._crop_probs:
            reward_scores = torch.clamp(reward_scores, min=0, max=1)

        if self._inverse:
            reward_scores = 1-reward_scores
        if self._method == "linear":
            return scores + reward_scores*self._beta
        else:
            raise ValueError(f"method {self._method} not supported")