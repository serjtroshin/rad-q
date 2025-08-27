import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm.auto import tqdm
import numpy as np


# takes in a EvalPrediction and returns a dictionary string to metric values.
def mse(eval_preds):
    predictions, labels = eval_preds    # (eval set size, 1)
    mse_metric = evaluate.load("mse")
    # flatten the predictions and labels
    # predictions = predictions.flatten()
    # labels = labels.flatten()
    if isinstance(predictions, tuple) and len(predictions) == 2:  # preds, cache
        predictions = predictions[0]
    predictions = predictions[:, 0]
    if len(labels.shape) == 2:
        labels = labels[:, 0]  # only eval toxicity
    return mse_metric.compute(predictions=predictions, references=labels)


def distinctness(generations):
    unigrams, bigrams, trigrams = set(), set(), set()
    total_words = 0
    
    for gen in generations:
        o = gen.split(' ')
        total_words += len(o)
        unigrams.update(o)
        for i in range(len(o) - 1):
            bigrams.add(o[i] + '_' + o[i + 1])
        for i in range(len(o) - 2):
            trigrams.add(o[i] + '_' + o[i + 1] + '_' + o[i + 2])
    
    return len(unigrams) / total_words, len(bigrams) / total_words, len(trigrams) / total_words


def compute_perplexity(args, generation, rad, device='cuda', use_model_itself=False):
    if use_model_itself:
        model = rad._lm
        tokenizer = rad._lm_tokenizer
        print("using model to evaluate perplexity")
    else:
        if "gpt2" in args.lm:
            print("using gpt2-xl to evaluate perplexty")
            model = AutoModelForCausalLM.from_pretrained('gpt2-xl', device_map=device)
            tokenizer = AutoTokenizer.from_pretrained('gpt2-xl')
        elif "use_llama_7b_ppl" in args.lm:
            print("using llama-7b to evaluate perplexity")
            # Llama-2-70b-hf
            model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', device_map=device, torch_dtype=torch.bfloat16)
            tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
        elif "llama" in args.lm or "Llama" in args.lm:
            print("using llama-13b to evaluate perplexity")
            # Llama-2-70b-hf
            model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-13b-hf', device_map=device, torch_dtype=torch.bfloat16)
            tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-13b-hf')
        elif "olmo" in args.lm:
            print("using olmo to evaluate perplexity")
            model = AutoModelForCausalLM.from_pretrained('allenai/OLMo-1B', device_map=device, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained('allenai/OLMo-1B', trust_remote_code=True)
        else:
            raise ValueError("Unknown LM model")
    
        
    perplexities = []
    logs = []
    
    pbar = tqdm(generation, total=len(generation), desc='Evaluate Fluency')
    for it, row in enumerate(pbar):
        prompt = row['prompt']['text']
        prompt_input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        with torch.inference_mode():
            prompt_loss = model(prompt_input_ids, labels=prompt_input_ids)[0]
            prompt_loss *= (prompt_input_ids.shape[1]-1)
            
            for cont in row['generations']:
                cont_text = cont['text']
                if "llama" in args.lm or "Llama" in args.lm:
                    full_text = prompt+" "+cont_text
                else:
                    full_text = prompt+cont_text
                full_input_ids = tokenizer.encode(prompt+cont_text, return_tensors='pt').to(device)
                full_loss = model(full_input_ids, labels=full_input_ids)[0] * (full_input_ids.shape[1]-1)
                loss = (full_loss - prompt_loss) / (full_input_ids.shape[1] - prompt_input_ids.shape[1])
                ppl = torch.exp(loss).item()
                
                if ppl < 1e5:
                    perplexities.append(ppl)
                logs.append({
                    'prompt': row['prompt'],
                    'cont': cont,
                    'ppl': ppl
                })
                    
                # if ppl > 100:
                if args.verbose:
                    print(f"ppl = {ppl:.3f}, prompt = '{prompt}', cont = '{cont_text}'")


        pbar.set_description(
            f'{it}: mean ppl = {np.mean(perplexities):.3f}'
        )
        
    if args.verbose:
        return perplexities, logs
    return perplexities