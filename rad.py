from transformers import (
    LogitsProcessorList,
    TopPLogitsWarper,
)
from utils.logits_processor import (
    RewardAugmentedLogitsProcessor,
    RewardAugmentedLogitsProcessorNoPkv,
    RewardAugmentedLogitsProcessorSameTokenizer
)
from utils.expert_logits_processor import ExternalExpertsLogitsProcessor

class RewardAugmentedDecoder():
    
    def __init__(self, language_model, lm_tokenizer, reward_model, rm_tokenizer, 
                 max_length, num_gpus=4, inverse=False, efficient=True):
        self._lm = language_model
        self._lm_tokenizer = lm_tokenizer
        self._rm = reward_model
        self._rm_tokenizer = rm_tokenizer
        self._max_length = max_length
        self._num_gpus = num_gpus
        self._inverse = inverse
        self._efficient = efficient

    def sample(
            self, 
            prompts,
            max_new_tokens=20,
            num_return_sequences=25, 
            method="linear",
            beta=30,
            data_container=None,
            generation_log=None,
            decoding_kwargs={},
            return_continuation_only=True,
        ):
        input_ids = self._lm_tokenizer.batch_encode_plus(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_length-max_new_tokens,
        ).to('cuda')
        
        # dry run
        if not self._rm:
            print("dry run", decoding_kwargs['top_p'], decoding_kwargs['topk'])
            outputs = self._lm.generate(
                **input_ids,
                # min_new_tokens=2,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                # temperature=0.7,
                top_p=decoding_kwargs['top_p'],
                top_k=decoding_kwargs['topk'],
                num_return_sequences=num_return_sequences,
            )
        else:
            if self._efficient:
                # if "gpt2" in self._lm_tokenizer.name_or_path:
                print("using our implementation of Logit Processor for the same lm and rm tokenizer (gpt2)")
                logits_processor = LogitsProcessorList([
                    # TopPLogitsWarper(top_p=0.9),
                    RewardAugmentedLogitsProcessorSameTokenizer(
                        self._lm_tokenizer,  # same as rm_tokenizer
                        self._rm_tokenizer,
                        self._rm,
                        topk=decoding_kwargs['topk'],
                        method=method,
                        beta=beta,
                        inverse=self._inverse,
                        data_container=data_container,
                        generation_log=generation_log,
                    ),
                ])
            else:
                print("using RewardAugmentedLogitsProcessorNoPkv")
                logits_processor = LogitsProcessorList([
                    # TopPLogitsWarper(top_p=0.9),
                    RewardAugmentedLogitsProcessorNoPkv(
                        self._lm_tokenizer,
                        self._rm_tokenizer,
                        self._rm,
                        topk=decoding_kwargs['topk'],
                        method=method,
                        beta=beta,
                        inverse=self._inverse,
                        generation_log=generation_log,
                    ),
                ])
                
            outputs = self._lm.generate(
                **input_ids,
                logits_processor=logits_processor,
                # min_new_tokens=2,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.0,
                # top_p=0.9,
                num_return_sequences=num_return_sequences,
            )
        
        
        if return_continuation_only:
            input_length = len(input_ids.input_ids[0])
            outputs = outputs[:, input_length:]          # remove prompt
            
        ret = self._lm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ret = [ret[i:i+num_return_sequences] for i in range(0, len(ret), num_return_sequences)]
        
        return ret


class RewardAugmentedDecoderRAW:
    
    def __init__(self, language_model, lm_tokenizer, max_length, num_gpus=4):
        self._lm = language_model
        self._lm_tokenizer = lm_tokenizer
        self._max_length = max_length
        self._num_gpus = num_gpus

    def sample(
            self, 
            prompts,
            max_new_tokens=20,
            num_return_sequences=25, 
            decoding_kwargs={},
            beta=None,
            return_continuation_only=True
        ):
        input_ids = self._lm_tokenizer.batch_encode_plus(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_length-max_new_tokens,
        ).to('cuda')

        outputs = self._lm.generate(
            **input_ids,
            # min_new_tokens=2,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_k=decoding_kwargs['topk'],
            top_p=decoding_kwargs['top_p'],
            num_return_sequences=num_return_sequences,
        )

        if return_continuation_only:
            input_length = len(input_ids.input_ids[0])
            outputs = outputs[:, input_length:]          # remove prompt

        ret = self._lm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ret = [ret[i:i+num_return_sequences] for i in range(0, len(ret), num_return_sequences)]
        
        return ret