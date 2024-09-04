import os
import pickle
import sys

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

sys.path.append('../')
import argparse
import pickle

import llama
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import (get_gemma_activations, get_llama_activations_bau,
                   tokenized_tqa, tokenized_tqa_gen, tokenized_tqa_gen_end_q)

HF_NAMES = {
    # 'llama_7B': 'baffo32/decapoda-research-llama-7B-hf',
    'llama_7B': 'huggyllama/llama-7b',
    'alpaca_7B': 'circulus/alpaca-7b', 
    'vicuna_7B': 'AlekseyKorshuk/vicuna-7b', 
    'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf', 
    'llama2_chat_13B': 'meta-llama/Llama-2-13b-chat-hf', 
    'llama2_chat_70B': 'meta-llama/Llama-2-70b-chat-hf', 
    'llama3_8B': 'meta-llama/Meta-Llama-3-8B',
    'llama3_8B_instruct': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'llama3_70B': 'meta-llama/Meta-Llama-3-70B',
    'llama3_70B_instruct': 'meta-llama/Meta-Llama-3-70B-Instruct',
    'gemma_2_2B': 'google/gemma-2-2b',
}

REPOS = {
    'nqopen' : "baonn/nqopen",
    'truthful_qa' : "truthful_qa",
    'trivia_qa': "baonn/trivia_qa"
}

def main(): 
    """
    Specify dataset name as the first command line argument. Current options are 
    "tqa_mc2", "piqa", "rte", "boolq", "copa". Gets activations for all prompts in the 
    validation set for the specified dataset on the last token for llama-7B. 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama_7B')
    parser.add_argument('--model_prefix', type=str, default='', help='prefix of model name')
    parser.add_argument('--dataset_name', type=str, default='tqa_mc2')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--train_dataset', type=str, default='truthful_qa', help='Dataset used for training')
    parser.add_argument("--model_dir", type=str, default=None, help='local directory with model data')
    args = parser.parse_args()

    model_name_or_path = HF_NAMES[args.model_prefix + args.model_name]

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if 'gemma' in model_name_or_path.lower():
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")

    device = "cuda"

    if args.dataset_name == "tqa_mc2":
        dataset = load_dataset(REPOS[args.train_dataset], "multiple_choice")['validation']
        formatter = tokenized_tqa
    elif args.dataset_name == "tqa_gen":
        dataset = load_dataset(REPOS[args.train_dataset], 'generation')['validation']
        formatter = tokenized_tqa_gen
    elif args.dataset_name == 'tqa_gen_end_q':
        dataset = load_dataset(REPOS[args.train_dataset], 'generation')['validation']
        formatter = tokenized_tqa_gen_end_q
    else:
        raise ValueError("Invalid dataset name")

    print("Tokenizing prompts")
    if args.dataset_name == "tqa_gen" or args.dataset_name == "tqa_gen_end_q": 
        prompts, labels, categories = formatter(dataset, tokenizer)
        with open(f'../features/{args.train_dataset}/{args.model_name}_{args.dataset_name}_categories.pkl', 'wb') as f:
            pickle.dump(categories, f)
    else: 
        prompts, labels = formatter(dataset, tokenizer)

    all_layer_wise_activations = []
    all_head_wise_activations = []

    print("Getting activations")
    for prompt in tqdm(prompts):
        if 'gemma' in model_name_or_path.lower():
            layer_wise_activations, head_wise_activations, _ = get_gemma_activations(
                model, prompt, device)
        else:
            layer_wise_activations, head_wise_activations, _ = get_llama_activations_bau(
                model, prompt, device)
        
        all_layer_wise_activations.append(layer_wise_activations[:,-1,:].copy())
        all_head_wise_activations.append(head_wise_activations[:,-1,:].copy())

    print("Saving labels")
    np.save(f'../features/{args.train_dataset}/{args.model_name}_{args.dataset_name}_labels.npy', labels)

    print("Saving layer wise activations")
    np.save(f'../features/{args.train_dataset}/{args.model_name}_{args.dataset_name}_layer_wise.npy', all_layer_wise_activations)
    
    print("Saving head wise activations")
    np.save(f'../features/{args.train_dataset}/{args.model_name}_{args.dataset_name}_head_wise.npy', all_head_wise_activations)

if __name__ == '__main__':
    main()
