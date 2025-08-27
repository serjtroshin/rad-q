import os
from pathlib import Path
import torch
from transformers import (
    set_seed,
    pipeline,
)
import argparse
from utils.parser_utils import load_rad
from utils.metrics import distinctness, compute_perplexity
import torch
from tqdm.auto import tqdm
import numpy as np
import json

def evaluate_model_on_dataset(args, rad, eval_prompts):

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    sentiment_scores = []
    positive_probs = []
    dist_n = []
    generation = []
    report = {}

    if args.test:
        eval_prompts = eval_prompts[:100]
        
    eval_prompt_chunks = list(chunks(eval_prompts, args.batch_size))

    pipe = pipeline('sentiment-analysis', device=0)     # model = 'distilbert-base-uncased-finetuned-sst-2-english'

    pbar = tqdm(eval_prompt_chunks)
    for chunk in pbar:
        # for debug:
        # chunk = ["Meanwhile the iron and lead"]
        with torch.inference_mode():
            # generated_texts: List[List[str]] (batch_size, num_return_sequences)
            generated_texts = rad.sample(
                chunk,
                max_new_tokens=args.max_new_tokens,
                beta=args.beta,
                num_return_sequences=args.num_return_sequences,
                decoding_kwargs={"topk": args.topk, "top_p": args.top_p},
            )
            
        for i, samples in enumerate(generated_texts):   
            # samples of a prompt: (num_return_sequences,)
            sentiment_score = pipe([chunk[i]+s for s in samples], truncation=True)
            sentiment_scores.append(sentiment_score)
            
            positive_proportion = sum([1 for s in sentiment_score if s['label'] == 'POSITIVE'])/len(sentiment_score)
            positive_probs.append(positive_proportion)
            
            dist_n.append(distinctness(samples))
            
            generation.append({
                'prompt': {"text": chunk[i]},
                'generations': 
                    [{"text": sp, "label": ss['label'], "score": ss['score']} for sp, ss in zip(samples, sentiment_score)]
            })

        pbar.set_description(
            f'positive rate = {"{:.3f}".format(np.mean(positive_probs))}, '\
            f'dist-n = {["{:.3f}".format(x) for x in np.nanmean(np.array(dist_n), axis=0)]}'
        )

    ppl = compute_perplexity(args, generation, rad)
    
    report.update({
        'positive_rate': np.mean(positive_probs),
        'dist_n': np.nanmean(np.array(dist_n), axis=0).tolist(),
        "perplexity": np.mean(ppl)
    })
    
    return report, generation

def load_dataset(args):
    prompts = []
    if args.dataset == 'negative':
        file_dir = "datasets/sentiment_prompts-10k/negative_prompts.jsonl"
    elif args.dataset == 'neutral':
        file_dir = "datasets/sentiment_prompts-10k/neutral_prompts.jsonl"
    elif args.dataset == 'positive':
        file_dir = "datasets/sentiment_prompts-10k/positive_prompts.jsonl"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    with open(file_dir) as f:
        for line in f:
            prompts.append(json.loads(line)['prompt']['text'])
    return prompts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="outputs/")
    parser.add_argument("--dataset", choices=['negative','neutral','positive'], default='negative')
    
    parser.add_argument("--beta", default=10, type=int)
    parser.add_argument("--topk", default=None, type=int)
    parser.add_argument("--top_p", default=None, type=float)
    parser.add_argument("--inverse", action="store_true")      # steer toward lower reward
    
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_return_sequences", default=25, type=int)
    parser.add_argument("--max_new_tokens", default=20, type=int)
    
    parser.add_argument("--lm", default="gpt2-large", choices=
        ["gpt2-large","gpt-neox-20b","Llama-2-7b-hf", "Llama-2-13b-hf", "Llama-2-70b-hf"])
    
    parser.add_argument("--rm", default=None, choices=["gpt2", "gpt2-soft", "gpt2-soft-distil", "gpt2-delta-distil", "TinyLlama-1.1B-intermediate-step-1431k-3T"])
    parser.add_argument("--not_efficient", action="store_false", dest="efficient")
    parser.add_argument("--rad_mode", default="soft", choices=["soft", "input", "raw"], help="soft: use ARM, input: use RAD, raw: use raw model")
    parser.add_argument("--limit_eval_samples", default=None, type=int)

    parser.add_argument("--rm_dir", default=None)
    
    parser.add_argument("--test", default=False, type=bool)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--name", default="sentiment")

    parser.add_argument("--lm_head_name", default="linear_with_baseline", type=str, choices=["linear", "linear_with_baseline"])
    parser.add_argument("--sigmoid", action="store_true")
    
    args = parser.parse_args()
    if args.topk == -1:
        args.topk = None
    return args


def main(args):
    set_seed(1)
    dataset = load_dataset(args)
    if args.limit_eval_samples:
        dataset = dataset[:args.limit_eval_samples]
    rad = load_rad(args, out_features=1, map_location=torch.device('cuda'))
    if args.rm_dir is not None:
        experiment=Path(args.rm_dir).parts[-2]
    else:
        experiment="GPT-2"
    print(experiment)
    results, generation = evaluate_model_on_dataset(args, rad, dataset)
    
    save_dir=Path(args.outdir) / f"{experiment}_{args.name}"
    save_dir.mkdir(parents=True, exist_ok=True)

    name=args.dataset
    if args.dataset == "neutral":
        name = f"neutral_inverse{args.inverse}"
    with open(
        save_dir / f'sentiment_{name}_report_{args.lm}_{args.rm}_top{args.topk if args.top_p is None else args.top_p}_beta{args.beta}_{args.dataset}_lim{args.limit_eval_samples}.json', 'w'
    ) as f:
        json.dump(results, f)
    
    with open(
        save_dir / f'sentiment_{name}_generation_{args.lm}_{args.rm}_top{args.topk if args.top_p is None else args.top_p}_beta{args.beta}_{args.dataset}_lim{args.limit_eval_samples}.jsonl', 'w'
    ) as f:
        for entry in generation:
            json.dump(entry, f)
            f.write("\n")


if __name__ == "__main__":
    args = parse_args()
    main(args)
