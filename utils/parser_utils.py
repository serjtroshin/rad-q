
import torch

# from peft import prepare_model_for_kbit_training, get_peft_model

from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizerFast

from reward_modeling.reward_model import GPT2RewardModel, GPT2RewardSoftModel
from reward_modeling.reward_model_llama import LlamaRewardModel, LlamaARMModel, LlamaDeltaDistillationModel
from reward_modeling.load import get_student_model_state_dict

from rad import RewardAugmentedDecoder, RewardAugmentedDecoderRAW
from rad_soft import RewardAugmentedSoftDecoder

from utils.utils import prepare_lm


def  load_gpt2_rm(args, out_features=7, map_location=torch.device('cuda')):
    if args.rm is None:
        rm = None
        rm_tokenizer = None
    elif args.rm == 'gpt2':
        rm_tokenizer = AutoTokenizer.from_pretrained(args.rm)
        rm_tokenizer.pad_token = rm_tokenizer.eos_token
        rm_tokenizer.padding_side = 'right'
        rm_tokenizer.max_length = 1024
        
        rm = GPT2RewardModel(reward_model_name=args.rm, out_features=out_features)
        
        state_dict = torch.load(args.rm_dir, map_location=map_location)
        rm.load_state_dict(state_dict)
        rm = rm.to(map_location)
    elif "gpt2-soft" in args.rm or "gpt2-delta" in args.rm:
        rm_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        rm_tokenizer.pad_token = rm_tokenizer.eos_token
        rm_tokenizer.padding_side = 'right'
        rm_tokenizer.max_length = 1024
        
        rm = GPT2RewardSoftModel(reward_model_name="gpt2", 
                                 out_features=out_features, 
                                 lm_head_name=args.lm_head_name,
                                 n_components=None,
                                 prepend_bos_token=False,  # we prepend in LogitsProcessor
                                 loss_fn=args.loss_fn,
                                 )
        
        state_dict = torch.load(args.rm_dir, map_location=map_location)
        if "distil" in args.rm:
            state_dict = get_student_model_state_dict(state_dict)
        # "model.lm_projection.weight" -> "head.lm_projection.weight"
        state_dict = {k.replace("model.lm_projection.weight", "head.lm_projection.weight"): v for k, v in state_dict.items()}
        # state_dict = get_student_model_state_dict(state_dict)
        rm.load_state_dict(state_dict)
        rm = rm.to(map_location)
    else:
        raise ValueError(f"Reward model {args.rm} not supported.")
    return rm, rm_tokenizer


def load_llama_rm(args, out_features=7, map_location=torch.device('cuda')):
    RM_LLAMA_MODEL = "meta-llama/Llama-2-7b-hf"
    if "llama" in args.rm or "Llama" in args.rm:
        rm_tokenizer = LlamaTokenizerFast.from_pretrained(RM_LLAMA_MODEL)
        rm_tokenizer.pad_token = rm_tokenizer.eos_token
        rm_tokenizer.padding_side = 'right'
        rm_tokenizer.max_length = 4096
        
        if args.model_type == "rad":
            rm = LlamaRewardModel(reward_model_name=f"TinyLlama/{args.rm}", out_features=out_features)
        else:
            rm = LlamaARMModel(
                reward_model_name=f"TinyLlama/{args.rm}", 
                out_features=out_features, 
                lm_head_name=args.lm_head_name,
            )
        
        if "lora" in args.rm_dir:
            raise NotImplementedError("LORA not supported for LLAMA")
        else:    
            print("loading non-LORA model")
            state_dict = torch.load(args.rm_dir, map_location=map_location)
            state_dict = get_student_model_state_dict(state_dict)
            rm.load_state_dict(state_dict)
        rm = rm.to(map_location)
    else:
        raise ValueError(f"Reward model {args.rm} not supported.")
    return rm, rm_tokenizer

def load_rad(args, out_features=7, map_location=torch.device('cuda')):
    if args.rm is None:
        rm, rm_tokenizer = None, None
        add_bos_token = False
    elif "gpt2" in args.rm:
        rm, rm_tokenizer = load_gpt2_rm(args, out_features, map_location)
        add_bos_token = True
    elif "llama" in args.rm or "Llama" in args.rm:
        rm, rm_tokenizer = load_llama_rm(args, out_features, map_location)
        add_bos_token = False
    else:
        raise ValueError(f"Reward model {args.rm} not supported.")
    
    lm, lm_tokenizer, max_length = prepare_lm(args.lm)
    
    if args.rad_mode == "soft":
        print("Using soft RAD")
        rad = RewardAugmentedSoftDecoder(
            lm, 
            lm_tokenizer, 
            rm, 
            max_length, 
            num_gpus=torch.cuda.device_count(),
            inverse=args.inverse,
            efficient=args.efficient,
            add_bos_token=add_bos_token
        )
    elif args.rad_mode == "input":
        print("Using hard RAD")
        rad = RewardAugmentedDecoder(
            lm, 
            lm_tokenizer, 
            rm, 
            rm_tokenizer, 
            max_length, 
            num_gpus=torch.cuda.device_count(),
            inverse=args.inverse,
            efficient=args.efficient
        )
    elif args.rad_mode == "raw":  # no reward model
        rad = RewardAugmentedDecoderRAW(
            lm, 
            lm_tokenizer, 
            max_length, 
            num_gpus=torch.cuda.device_count(),
        )
    return rad