from transformers import (
    LogitsProcessorList,
    TopPLogitsWarper,
)
from utils.logits_processor_soft import (
    RewardAugmentedLogitsSoftProcessor,
    RewardAugmentedLogitsSoftProcessorNoPkv,
    RewardAugmentedLogitsSoftProcessorOrig,
    parse_decoding_kwargs
)

class RewardAugmentedSoftDecoder():
    def __init__(self, language_model, lm_tokenizer, reward_model, 
                 max_length, num_gpus=4, inverse=False, efficient=True, add_bos_token=True, logit_processor_kwargs={}
                 ):
        self._lm = language_model
        self._lm_tokenizer = lm_tokenizer
        self._rm = reward_model
        # self._rm_tokenizer = rm_tokenizer  # must be the same as lm_tokenizer
        self._max_length = max_length
        self._num_gpus = num_gpus
        self._inverse = inverse
        self._efficient = efficient
        self._logit_processor_kwargs = logit_processor_kwargs
        self._add_bos_token = add_bos_token

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
            outputs = self._lm.generate(
                **input_ids,
                # min_new_tokens=2,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.0,
                # top_p=0.9,
                top_k=decoding_kwargs['topk'],
                num_return_sequences=num_return_sequences,
            )
        else:
            decoding_logit_processor = parse_decoding_kwargs(decoding_kwargs)
            if self._efficient:
                logits_processor = LogitsProcessorList([
                    decoding_logit_processor,
                    RewardAugmentedLogitsSoftProcessor(
                        self._lm_tokenizer,
                #         self._rm_tokenizer,
                        self._rm,
                        topk=-1 if decoding_kwargs['topk'] != self._lm_tokenizer.vocab_size  else self._lm_tokenizer.vocab_size,
                        method=method,
                        beta=beta,
                        inverse=self._inverse,
                        data_container=data_container,
                        generation_log=generation_log,
                        prepend_bos=self._add_bos_token,
                        **self._logit_processor_kwargs
                    ),
                ])
            else:
                logits_processor = LogitsProcessorList([
                    # TopPLogitsWarper(top_p=0.9),
                    RewardAugmentedLogitsSoftProcessorNoPkv(
                        self._lm_tokenizer,
                        # self._rm_tokenizer,
                        self._rm,
                        topk=decoding_kwargs['topk'],
                        method=method,
                        beta=beta,
                        inverse=self._inverse,
                        generation_log=generation_log,
                        **self._logit_processor_kwargs
                    ),
                ])
            print(f"Using {logits_processor}")
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
