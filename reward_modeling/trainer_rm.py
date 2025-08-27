import itertools
from pathlib import Path
from typing import List
from datasets import load_dataset, concatenate_datasets, Value
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer, set_seed
from transformers.trainer import get_parameter_names, ALL_LAYERNORM_LAYERS
import numpy as np
import torch.nn as nn
import torch
import argparse
import random
from utils.utils import read_yamls, _strtobool, get_dataset, get_rm_tokenizer, get_reward_model
from utils.metrics import mse
import os
from torch.utils.data import Subset
from tqdm import tqdm
from transformers.training_args import OptimizerNames


def argument_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--wandb_entity", type=str, default="istar98")
    parser.add_argument("--experiment", type=str, default="default")
    parser.add_argument("--resume_from_checkpoint", action="store_true", help="Resume from last saved checkpoint")
    parser.add_argument("--init_model_weights", type=str, default=None)
    parser.add_argument("--use_data_subsampling", action="store_true", help="Subsample data randomly")
    parser.add_argument("--retokenize", action="store_true", help="Retokenize the dataset")
    parser.add_argument("--total_batch_size", default=100, type=int)

    args, remaining = parser.parse_known_args()

    # Config from YAML
    conf = {}
    configs = read_yamls("./configs")
    for name in args.configs:
        if "," in name:
            for n in name.split(","):
                conf.update(configs[n])
        else:
            conf.update(configs[name])

    conf["wandb_entity"] = args.wandb_entity
    conf["wandb_name"] = args.experiment
    conf["resume_from_checkpoint"] = args.resume_from_checkpoint
    conf["init_model_weights"] = args.init_model_weights
    conf["use_data_subsampling"] = args.use_data_subsampling
    conf["retokenize"] = args.retokenize
    conf["total_batch_size"] = args.total_batch_size

    if "n_components" in conf:
        conf["wandb_name"] += f"_ncomp{conf['n_components']}"

    # Override config from command-line
    parser = argparse.ArgumentParser()
    for key, value in conf.items():
        type_ = type(value) if value is not None else str
        if type_ == bool:
            type_ = _strtobool
        parser.add_argument(f"--{key}", type=type_, default=value)

    return parser.parse_args(remaining)

def main():
    training_args = argument_parsing()
    print(training_args)
    training_args.output_dir = Path(training_args.output_dir) / training_args.wandb_name
    print("experiment dir", training_args.output_dir)
    n_gpu = torch.cuda.device_count()
    total_batch_size=training_args.total_batch_size
    training_args.gradient_accumulation_steps = total_batch_size // training_args.per_device_train_batch_size // n_gpu
    print("accumulation steps", training_args.gradient_accumulation_steps)
    assert n_gpu * training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps == total_batch_size
    set_seed(training_args.seed)

    model = get_reward_model(training_args)
    tokenizer = get_rm_tokenizer(training_args)
    train, evals = get_dataset(training_args, tokenizer, retokenize=training_args.retokenize)

    data_collator = DataCollatorWithPadding(
        tokenizer,
        padding=True,
        max_length=training_args.max_length,
    )

    if training_args.verbose:
        print("Dataset stats before sampling:")
        total = len(train)
        for d in itertools.chain(train.datasets, evals.values()):
            if isinstance(d, Subset):
                name = f"Subset of {type(d.dataset).__name__}"
                if hasattr(d.dataset, "name"):
                    name += f" ({d.dataset.name})"
            else:
                name = type(d).__name__
                if hasattr(d, "name"):
                    name += f" ({d.name})"
            print(f"{name}: {len(d)} ({len(d) / total:%})")
        print(f"Total train: {total}")

    optimizer = OptimizerNames.ADAMW_HF

    output_dir = (
        training_args.output_dir
        if training_args.output_dir
        else f"{training_args.reward_model_name}-{training_args.dataset}-finetuned"
    )

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_args.num_train_epochs,
        warmup_steps=training_args.warmup_steps,
        learning_rate=float(training_args.learning_rate),
        optim=optimizer,
        fp16=training_args.dtype in ["fp16", "float16"],
        bf16=training_args.dtype in ["bf16", "bfloat16"],
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        adam_beta1=training_args.adam_beta1,
        adam_beta2=training_args.adam_beta2,
        adam_epsilon=float(training_args.adam_epsilon),
        weight_decay=training_args.weight_decay,
        max_grad_norm=training_args.max_grad_norm,
        logging_steps=training_args.logging_steps,
        save_total_limit=training_args.save_total_limit,
        evaluation_strategy="steps",
        eval_steps=training_args.eval_steps,
        save_strategy=training_args.save_strategy,
        save_steps=training_args.save_steps,
        resume_from_checkpoint=training_args.resume_from_checkpoint,
        report_to="wandb" if training_args.log_wandb else None,
        save_safetensors=training_args.save_safetensors,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
    )

    if not training_args.log_wandb:
        os.environ["WANDB_MODE"] = "offline"

    if training_args.log_wandb:
        import wandb

        wandb.init(
            project="reward-model",
            entity=training_args.wandb_entity,
            resume=training_args.resume_from_checkpoint,
            name=f"{training_args.reward_model_name}-rm-{training_args.wandb_name}",
            config=training_args,
        )
        
    compute_metrics = mse
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=evals,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()