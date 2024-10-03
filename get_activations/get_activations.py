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
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
from lofit_models.modeling_llama import LlamaModel,LlamaForCausalLM
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
    'llama_7B_lofit_fold_0': 'huggyllama/llama-7b',
    'llama_7B_lofit_fold_1': 'huggyllama/llama-7b',
    'llama2_chat_13B_lofit_fold_0': 'meta-llama/Llama-2-13b-chat-hf',
    'llama2_chat_13B_lofit_fold_1': 'meta-llama/Llama-2-13b-chat-hf',
    'llama3_8B_lofit_fold_0': 'meta-llama/Meta-Llama-3-8B',
    'llama3_8B_lofit_fold_1': 'meta-llama/Meta-Llama-3-8B',
}
ADAPTERS = {
    'llama_7B_lofit_fold_0': '/home/users/nus/binhnt/scratch/baonn/lofit/saved_model/llama2_7B_truthfulqa_42_fold_0',
    'llama_7B_lofit_fold_1': '/home/users/nus/binhnt/scratch/baonn/lofit/saved_model/llama2_7B_truthfulqa_42_fold_1',
    'llama2_chat_13B_lofit_fold_0': '/home/users/nus/binhnt/scratch/baonn/lofit/saved_model/llama2_13B_truthfulqa_42_fold_0',
    'llama2_chat_13B_lofit_fold_1': '/home/users/nus/binhnt/scratch/baonn/lofit/saved_model/llama2_13B_truthfulqa_42_fold_1',
    'llama3_8B_lofit_fold_0': '/home/users/nus/binhnt/scratch/baonn/lofit/saved_model/llama3_8B_truthfulqa_42_fold_0',
    'llama3_8B_lofit_fold_1': '/home/users/nus/binhnt/scratch/baonn/lofit/saved_model/llama3_8B_truthfulqa_42_fold_1',
}

REPOS = {
    'nqopen' : "baonn/nqopen",
    'truthful_qa' : "truthful_qa",
    'trivia_qa': "baonn/trivia_qa"
}

def load_attention_components(model, path_A, path_v):
    # Load the saved parameters
    attn_A_params = torch.load(path_A)
    attn_v_params = torch.load(path_v)
    
    for i in range(model.config.num_hidden_layers):
        # Load attention A components back into the model
        attn_A = model.model.layers[i].self_attn.attn_A
        for j, module in enumerate(attn_A):
            module.data.copy_(attn_A_params[f'layer_{i}'][f'head_{j}'])
        
        # Load attention v components back into the model
        attn_v = model.model.layers[i].self_attn.attn_v
        for j, module in enumerate(attn_v):
            module.data.copy_(attn_v_params[f'layer_{i}'][f'head_{j}'])

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
    parser.add_argument('--cache_dir', type=str, default="", help="hugging face hub")
    args = parser.parse_args()
    logging.set_verbosity_error()
    model_name_or_path = HF_NAMES[args.model_prefix + args.model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if 'gemma' in model_name_or_path.lower():
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
    elif 'lofit' in (args.model_prefix + args.model_name).lower():
        if '13b' in args.model_name.lower():
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32

        model = LlamaForCausalLM.custom_from_pretrained(model_name_or_path, 
                                                device_map="auto",
                                                cache_dir=args.cache_dir,
                                                applied_module = "attention",
                                                applied_layers = None,
                                                torch_dtype=torch_dtype)
        load_attention_components(model, os.path.join(ADAPTERS[(args.model_prefix + args.model_name)], "A.pth"), os.path.join(ADAPTERS[(args.model_prefix + args.model_name)], "v.pth"))
        for param in model.parameters():
            param.requires_grad = False
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")

    device = "cuda"
    model = model.cuda()
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
