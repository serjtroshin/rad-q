from functools import partial
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    set_seed,
    pipeline,
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
)
from tokenizers.processors import TemplateProcessing
from datasets import load_dataset, Dataset, DatasetDict
from torch import nn
import torch
from pathlib import Path
import yaml
import random
import copy
from torch.utils.data import ConcatDataset, Subset
from reward_modeling.reward_model import GPT2RewardModel, GPT2RewardSoftModel, GPT2LMHeadModel, DistillationModel
from reward_modeling.reward_model_baseline_diff import DeltaDistillationModel
from reward_modeling.reward_model_llama import LlamaRewardModel, LlamaARMModel, LlamaDeltaDistillationModel, LlamaDistillationModel
from distutils.util import strtobool
import json

import pandas as pd
import pyarrow as pa

def get_dataset_name_and_kwargs_from_data_config(data_config):
    if isinstance(data_config, dict):
        name = list(data_config.keys())[0]

        # first copy the dict, then remove the size and fraction
        kwargs = copy.deepcopy(data_config[name])

        kwargs.pop("fraction", None)
        kwargs.pop("size", None)
        return name, kwargs
    else:
        return data_config, {}


def get_dataset(
    args,
    tokenizer,
    retokenize=False
) -> tuple[ConcatDataset, dict[str, Subset]]:
    train_datasets, evals = [], {}

    for data_config in args.datasets:
        dataset_name, kwargs = get_dataset_name_and_kwargs_from_data_config(data_config)
        train, val = get_one_dataset(args, dataset_name, tokenizer, retokenize=retokenize)

        if args.use_data_subsampling:
            # Subsample data (0.5) randomly
            train = train.train_test_split(test_size=0.5, seed=args.seed)["train"]
            print("Subsampled train dataset to 50%")
            
        train_datasets.append(train)

        if val is not None:
            evals[dataset_name] = Subset(val, list(range(min(len(val), args.eval_size)))) if args.eval_size else val

    train = ConcatDataset(train_datasets)
    return train, evals


def get_one_dataset(
    args, 
    dataset_name, 
    tokenizer,
    retokenize=False
):
    if "gpt2" in args.reward_model_name:
        model_type = ""
        if "soft" in args.reward_model_name:
            model_type = "soft"  # different max_length to add bos token
            assert tokenizer.max_length == 1023, "tokenizer max_length must be 1023 for soft model"
            args.max_length = 1023
    else:
        # assert args.max_length == 4096, "tokenizer max_length must be 4096 for llama models"
        model_type = args.reward_model_name.split("/")[-1]
    print(model_type, tokenizer.max_length, args.max_length)

    if dataset_name == "sst2":
        dataset = load_dataset("sst2")
        print("dataset", dataset)
        dataset = dataset.rename_columns({"label": "labels", "sentence": "text"})
        
        columns = dataset['train'].column_names
        columns_to_keep = ["text", "labels"]
        dataset = dataset.remove_columns(list(set(columns)-set(columns_to_keep)))
        
        def tokenize_dataset(examples):
            # remove the space at the end of each sentence
            return tokenizer([e[:-1] for e in examples["text"]], truncation=True, max_length=args.max_length)
        
        dataset = dataset.map(tokenize_dataset, batched=True,
            load_from_cache_file=not retokenize,
            cache_file_names={
                    'train': f"{args.sst2_dir}/train.tok{model_type}.arrow",
                    'test': f"{args.sst2_dir}/test.tok{model_type}.arrow",
                    'validation': f"{args.sst2_dir}/validation.tok{model_type}.arrow"
                }
            )
        train, eval = dataset['train'], dataset['validation']
        
    elif dataset_name == "amazon_polarity":
        dataset = load_dataset("amazon_polarity")
        print("dataset", dataset)
        dataset = dataset.rename_columns({"label": "labels", "content": "text"})
        
        columns = dataset['train'].column_names
        columns_to_keep = ["text", "labels"]
        dataset = dataset.remove_columns(list(set(columns)-set(columns_to_keep)))
        
        def tokenize_dataset(examples):
            return tokenizer(examples["text"], truncation=True, max_length=args.max_length)
        
        dataset = dataset.map(tokenize_dataset, batched=True,
            load_from_cache_file=not retokenize,
            cache_file_names={
                'train': f"{args.amazon_dir}/train.tok{model_type}.arrow", 
                'test': f"{args.amazon_dir}/test.tok{model_type}.arrow",
                'validation': f"{args.amazon_dir}/validation.tok{model_type}.arrow"
            }
        )
        train, eval = dataset['train'], dataset['test']
    
    elif dataset_name == "jigsaw_unintended_bias":
        dataset = load_dataset("jigsaw_unintended_bias", data_dir=args.jigsaw_dir, cache_dir=".cache/huggingface/jigsaw_unintended_bias/")
        columns = dataset['train'].column_names
        columns_to_keep = [
            "comment_text", "target", "severe_toxicity", "obscene",
            "identity_attack", "insult", "threat", "sexual_explicit"
        ]
        dataset = dataset.remove_columns(list(set(columns)-set(columns_to_keep)))
        dataset = dataset.map(
            lambda example: {"labels": [example["target"],
                                        example["severe_toxicity"],
                                        example["obscene"],
                                        example["identity_attack"],
                                        example["insult"],
                                        example["threat"],
                                        example["sexual_explicit"]]},
            remove_columns=columns_to_keep[1:],  # keep "comment_text" and "labels" only
            load_from_cache_file=True,
            cache_file_names={
                'train' : f"{args.jigsaw_dir}/train.arrow",
                'test_private_leaderboard': f"{args.jigsaw_dir}/test_private_leaderboard.arrow",
                'test_public_leaderboard': f"{args.jigsaw_dir}/test_public_leaderboard.arrow"
            }
        )
        dataset = dataset.rename_columns({"comment_text": "text"})
            
        def tokenize_dataset(examples):
            return tokenizer(examples["text"], truncation=True, max_length=args.max_length)
        
        dataset = dataset.map(tokenize_dataset, batched=True, load_from_cache_file=not retokenize,
                              cache_file_names={
                                'train' : f"{args.jigsaw_dir}/train.tok{model_type}.arrow",
                                'test_private_leaderboard': f"{args.jigsaw_dir}/test_private_leaderboard.tok{model_type}.arrow",
                                'test_public_leaderboard': f"{args.jigsaw_dir}/test_public_leaderboard.tok{model_type}.arrow"
                            }
        )
        train, eval = dataset['train'], dataset['test_public_leaderboard']

    elif dataset_name == "helpsteer":
        # nvidia/HelpSteer
        # using helpfulness as a metric
        dataset = load_dataset("nvidia/HelpSteer", cache_dir=".cache/huggingface/HelpSteer/")
        print("dataset", dataset)
        columns = dataset['train'].column_names
        print("columns", columns)

        dataset = {
            "train": dataset['train'],
            "test": dataset['validation'],
        }
        dataset = DatasetDict(dataset)
        # features: ['prompt', 'response', 'helpfulness', 'correctness', 'coherence', 'complexity', 'verbosity'],
        dataset = dataset.rename_columns({
            "helpfulness": "labels", 
            "prompt": "prompt",
            "response": "response",
            "correctness": "correctness",
            "coherence": "coherence",
            "complexity": "complexity",
            "verbosity": "verbosity"
        })
        # create text input simply as "prompt"\n\n"response"
        def tokenize_dataset(examples, mode="response"):
            if mode == "response":
                return tokenizer(examples["response"], truncation=True, max_length=args.max_length)
            elif mode == "prompt":
                o = tokenizer(examples["prompt"], truncation=True, max_length=args.max_length)
                o['prompt_ids'] = o['input_ids']
                o['attention_mask_prompt'] = o['attention_mask']
                del o['input_ids']
                del o['attention_mask']
            return o
        
        for mode in ["response", "prompt"]:
            dataset = dataset.map(partial(tokenize_dataset, mode=mode), batched=True,
                load_from_cache_file=not retokenize,
                cache_file_names={
                    'train': f"{args.helpsteer_dir}/train.tok{model_type}{mode}.arrow",
                    'test': f"{args.helpsteer_dir}/test.tok{model_type}{mode}.arrow",
                }
            )
        train, eval = dataset['train'], dataset['test']

    elif dataset_name == "beaver":
        # PKU-Alignment/BeaverTails
        dataset = load_dataset("PKU-Alignment/BeaverTails", cache_dir=".cache/huggingface/BeaverTails/")
        # rename "30k_train" to "train" and "30k_test" to "test"
        dataset = {'train': dataset['30k_train'], 'test': dataset['30k_test']}
        dataset = DatasetDict(dataset)
        print("dataset", dataset)
        columns = dataset['train'].column_names
        print("columns", columns)

        # labels are "is_safe" column
        dataset = dataset.rename_columns({"is_safe": "labels", "prompt": "prompt", "response": "response", "category": "category"})

        def tokenize_dataset(examples, mode="response"):
            if mode == "response":
                return tokenizer(examples["response"], truncation=True, max_length=args.max_length)
            elif mode == "prompt":
                o = tokenizer(examples["prompt"], truncation=True, max_length=args.max_length)
                o['prompt_ids'] = o['input_ids']
                o['attention_mask_prompt'] = o['attention_mask']
                del o['input_ids']
                del o['attention_mask']
            return o
        
        for mode in ["response", "prompt"]:
            dataset = dataset.map(partial(tokenize_dataset, mode=mode), batched=True,
                load_from_cache_file=not retokenize,
                cache_file_names={
                    'train': f"{args.beaver_dir}/train.tok{model_type}{mode}.arrow",
                    'test': f"{args.beaver_dir}/test.tok{model_type}{mode}.arrow",
                }
        )
        train, eval = dataset['train'], dataset['test']

    elif dataset_name == "id_full_rank":
        # create synthetic dataset of all pairs of tokens

        vocab_size = tokenizer.vocab_size
        RANK = args.test_rank  # should be more than model dim

        # create x, y as meshgrid of range(RANK), range(RANK)
        x = torch.arange(RANK)
        y = torch.arange(RANK)
        X, Y = torch.meshgrid(x, y)
        X = X.flatten()
        Y = Y.flatten()
        XY_cat = torch.cat([X.unsqueeze(1), Y.unsqueeze(1)], dim=1)
        R = (X == Y).float()
        # create pyarrow table with columns x, y, R
        train = Dataset.from_dict(
            {"input_ids": XY_cat, "labels": R}
        )
        print(train)
        eval = train

    elif dataset_name == "upper_triangular":
        # create synthetic dataset of all pairs of tokens

        vocab_size = tokenizer.vocab_size
        RANK = args.test_rank

        # create x, y as meshgrid of range(RANK), range(RANK)
        x = torch.arange(RANK)
        y = torch.arange(RANK)
        X, Y = torch.meshgrid(x, y)
        X = X.flatten()
        Y = Y.flatten()
        XY_cat = torch.cat([X.unsqueeze(1), Y.unsqueeze(1)], dim=1)
        R = (X == Y).float()

        XY_cat_lower_part = XY_cat[XY_cat[:, 0] >= XY_cat[:, 1]]
        R_lower_part = R[XY_cat[:, 0] >= XY_cat[:, 1]]
        # leave only lower triangular part where X <= Y

        # create pyarrow table with columns x, y, R
        train = Dataset.from_dict(
            {"input_ids": XY_cat_lower_part, "labels": R_lower_part}
        )
        print(train)
        eval = train

    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    
        
    return train, eval


def prepare_lm(model_name):

    if model_name == "gpt2-large":
        lm_tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
        lm = AutoModelForCausalLM.from_pretrained("gpt2-large", device_map='balanced_low_0')
        max_length = 1024
    elif model_name == "gpt2":
        lm_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        lm = AutoModelForCausalLM.from_pretrained("gpt2", device_map='balanced_low_0')
        max_length = 1024

    elif "llama" in model_name or "Llama" in model_name:
        model_name = f"meta-llama/{model_name}"
        lm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        lm = LlamaForCausalLM.from_pretrained(
            model_name, device_map='balanced_low_0', torch_dtype=torch.bfloat16)
        max_length = 4096

    else:
        raise ValueError(f"Model {model_name} not supported.")
        
    # set pad_token_id to eos_token_id because GPT2/Llama does not have a PAD token
    lm_tokenizer.pad_token = lm_tokenizer.eos_token
    lm_tokenizer.padding_side = 'left'                  # left padding while generating
    
    return lm, lm_tokenizer, max_length 


def get_gpt_rm_tokenizer(args):
    if "gpt2" in args.reward_model_name:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.reward_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    if 'soft' in args.reward_model_name:
        tokenizer.max_length = args.max_length - 1  # need prepend bos token
    else:
        tokenizer.max_length = args.max_length
    return tokenizer

def get_llama_rm_tokenizer(args):
    if "TinyLlama" in args.reward_model_name:
        tokenizer = AutoTokenizer.from_pretrained(f"TinyLlama/{args.reward_model_name}")
    elif "Llama" in args.reward_model_name:
        tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/{args.reward_model_name}")
    else:
        raise ValueError(f"Reward model {args.reward_model_name} not supported.")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    tokenizer.max_length = args.max_length
    return tokenizer

def get_rm_tokenizer(args):
    if "gpt2" in args.reward_model_name:
        tokenizer = get_gpt_rm_tokenizer(args)
    elif "llama" in args.reward_model_name or "Llama" in args.reward_model_name:
        tokenizer = get_llama_rm_tokenizer(args)
    else:
        raise ValueError(f"Reward model {args.reward_model_name} not supported.")
    return tokenizer

def get_gpt_reward_model(args):
    if "gpt2-delta-distil" == args.reward_model_name:
        reward_model_name = "gpt2"

        # finetuned classifier expert
        teacher = GPT2RewardModel(reward_model_name=reward_model_name, out_features=args.out_features)
        state_dict = torch.load(args.teacher_path)
        teacher.load_state_dict(state_dict)
        teacher.eval()

        student = GPT2RewardSoftModel(
            reward_model_name=reward_model_name,
            out_features=args.out_features, 
            loss_fn=args.loss_fn,
            lm_head_name=args.lm_head_name,
            freeze_embeddings=args.freeze_embeddings,
            freeze_rm=args.freeze_rm,
            embeddings_reg_loss=args.embeddings_reg_loss,
            n_components=args.n_components if hasattr(args, "n_components") else None,
            project_emb_on_sphere=args.normalize_embeddings,
            prepend_bos_token=args.prepend_bos_token,
            bos_token_id=AutoTokenizer.from_pretrained("gpt2").bos_token_id,
        )
        model = DeltaDistillationModel(
            teacher, student, regularization=args.regularization if hasattr(args, "regularization") else None
        )
        if args.init_model_weights is not None:
            print("loading model weights from", args.init_model_weights)
            state_dict = torch.load(args.init_model_weights)
            model.load_state_dict(state_dict)

    elif "gpt2-soft-distil" == args.reward_model_name:
        reward_model_name = "gpt2"

        # finetuned classifier expert
        teacher = GPT2RewardModel(reward_model_name=reward_model_name, out_features=args.out_features)
        state_dict = torch.load(args.teacher_path)
        teacher.load_state_dict(state_dict)
        teacher.eval()

        student = GPT2RewardSoftModel(
            reward_model_name=reward_model_name,
            out_features=args.out_features, 
            loss_fn=args.loss_fn,
            lm_head_name=args.lm_head_name,
            freeze_embeddings=args.freeze_embeddings,
            freeze_rm=args.freeze_rm,
            embeddings_reg_loss=args.embeddings_reg_loss,
            n_components=args.n_components if hasattr(args, "n_components") else None,
            project_emb_on_sphere=args.normalize_embeddings,
            prepend_bos_token=args.prepend_bos_token,
            bos_token_id=AutoTokenizer.from_pretrained("gpt2").bos_token_id,
        )
        model = DistillationModel(
            teacher, student, regularization=args.regularization if hasattr(args, "regularization") else None
        )
        if args.init_model_weights is not None:
            print("loading model weights from", args.init_model_weights)
            state_dict = torch.load(args.init_model_weights)
            model.load_state_dict(state_dict)
    elif "gpt2-soft" == args.reward_model_name:
        reward_model_name = "gpt2"
        model = GPT2RewardSoftModel(
            reward_model_name=reward_model_name,
            out_features=args.out_features, 
            loss_fn=args.loss_fn,
            lm_head_name=args.lm_head_name,
            freeze_embeddings=args.freeze_embeddings,
            freeze_rm=args.freeze_rm,
            embeddings_reg_loss=args.embeddings_reg_loss,
            n_components=args.n_components if hasattr(args, "n_components") else None,
            project_emb_on_sphere=args.normalize_embeddings,
            regularization=args.regularization if hasattr(args, "regularization") else None,
            prepend_bos_token=args.prepend_bos_token,
            bos_token_id=AutoTokenizer.from_pretrained("gpt2").bos_token_id,
        )
        if args.init_model_weights is not None:
            print("loading model weights from", args.init_model_weights)
            state_dict = torch.load(args.init_model_weights)
            model.load_state_dict(state_dict)
    elif "gpt2" in args.reward_model_name:
        model = GPT2RewardModel(
            reward_model_name=args.reward_model_name,
            out_features=args.out_features, 
            loss_fn=args.loss_fn
        )
    else:
        raise ValueError(f"Reward model {args.reward_model_name} not supported.")
    return model


def get_llama_reward_model(args):
    model = None
    if "TinyLlama" in args.reward_model_name:
        if args.distil_into_arm:
            student = LlamaARMModel(
                reward_model_name=f"TinyLlama/{args.reward_model_name}",
                out_features=args.out_features,
                loss_fn=args.loss_fn,
                lm_head_name=args.lm_head_name,
            )
            teacher = LlamaRewardModel(
                reward_model_name=f"TinyLlama/{args.reward_model_name}",
                out_features=args.out_features,
                loss_fn=args.loss_fn
            )
            state_dict = torch.load(args.teacher_path)
            teacher.load_state_dict(state_dict)
            teacher.eval()
            if args.distil_class == "orig_distil":
                model = LlamaDistillationModel(
                    teacher, student, regularization=args.regularization if hasattr(args, "regularization") else None
                )
            elif args.distil_class == "delta_distil":
                model = LlamaDeltaDistillationModel(
                    teacher, student, regularization=args.regularization if hasattr(args, "regularization") else None
                )
            else:
                raise ValueError(f"Distillation class {args.distil_class} not supported.")
            print("Using: ", model)
           
        else:
            model = LlamaRewardModel(
                reward_model_name=f"TinyLlama/{args.reward_model_name}",
                out_features=args.out_features,
                loss_fn=args.loss_fn
            )
    else:
        raise ValueError(f"Reward model {args.reward_model_name} not supported.")
        
    print("model", model)

    return model

def get_reward_model(args):
    if "gpt2" in args.reward_model_name:
        model = get_gpt_reward_model(args)
    elif "llama" in args.reward_model_name or "Llama" in args.reward_model_name:
        model = get_llama_reward_model(args)
    else:
        raise ValueError(f"Reward model {args.reward_model_name} not supported.")
    return model


def _strtobool(x):
    return bool(strtobool(x))


def read_yamls(dir):
    args = {}
    no_conf = True

    for config_file in Path(dir).glob("**/*.yaml"):
        no_conf = False
        with config_file.open("r") as f:
            args.update(yaml.safe_load(f))

    if no_conf:
        print(f"WARNING: No yaml files found in {dir}")

    return args