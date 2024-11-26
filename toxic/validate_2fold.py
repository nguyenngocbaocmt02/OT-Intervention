import torch
from einops import rearrange
import numpy as np
import pickle
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig, logging
from truthfulqa import metrics, models, utilities
from truthfulqa.configs import ANSWER_COL, BEST_COL, INCORRECT_COL
from truthfulqa.utilities import (find_start, format_best, format_prompt,
                                  format_prompt_with_answer_strings,
                                  split_multi_answer)

import sys
sys.path.append('../')
from lofit_models.modeling_llama import LlamaModel, LlamaForCausalLM
from utils import alt_tqa_evaluate, flattened_idx_to_layer_head, layer_head_to_flattened_idx, get_interventions_dict, get_top_heads, get_separated_activations, get_com_directions, get_ot_interventions_dict, alt_completion_evaluate
import llama

HF_NAMES = {
    # Base models
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
    'gpt2_large': 'openai-community/gpt2-large',
    # HF edited models (ITI baked-in)
    'honest_llama_7B': 'jujipotle/honest_llama_7B', # Heads=48, alpha=15
    # 'honest_llama2_chat_7B': 'likenneth/honest_llama2_chat_7B', # Heads=?, alpha=?
    'honest_llama2_chat_7B': 'jujipotle/honest_llama2_chat_7B', # Heads=48, alpha=15
    'honest_llama2_chat_13B': 'jujipotle/honest_llama2_chat_13B', # Heads=48, alpha=15
    'honest_llama2_chat_70B': 'jujipotle/honest_llama2_chat_70B', # Heads=48, alpha=15
    'honest_llama3_8B_instruct': 'jujipotle/honest_llama3_8B_instruct', # Heads=48, alpha=15
    'honest_llama3_70B_instruct': 'jujipotle/honest_llama3_70B_instruct', # Heads=48, alpha=15
    # Locally edited models (ITI baked-in)
    'local_llama_7B': 'results_dump/edited_models_dump/llama_7B_seed_42_top_48_heads_alpha_15',
    'local_llama2_chat_7B': 'results_dump/edited_models_dump/llama2_chat_7B_seed_42_top_48_heads_alpha_15',
    'local_llama2_chat_13B': 'results_dump/edited_models_dump/llama2_chat_13B_seed_42_top_48_heads_alpha_15',
    'local_llama2_chat_70B': 'results_dump/edited_models_dump/llama2_chat_70B_seed_42_top_48_heads_alpha_15',
    'local_llama3_8B_instruct': 'results_dump/edited_models_dump/llama3_8B_instruct_seed_42_top_48_heads_alpha_15',
    'local_llama3_70B_instruct': 'results_dump/edited_models_dump/llama3_70B_instruct_seed_42_top_48_heads_alpha_15',
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

PATHs = {
    'toxic': "../Toxic/train.csv"
}

def read_df(train_dataset):
    df = pd.read_csv(PATHs[train_dataset])
    return df

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='gpt2_large', choices=HF_NAMES.keys(), help='model name')
    parser.add_argument('--dataset_name', type=str, default='tqa_mc2', help='feature bank for training probes')
    parser.add_argument('--activations_dataset', type=str, default='tqa_mc2', help='feature bank for calculating std along direction')
    parser.add_argument('--num_heads', type=int, default=48, help='K, number of top heads to intervene on')
    parser.add_argument('--alpha', type=float, default=15, help='alpha, intervention strength')
    parser.add_argument("--num_fold", type=int, default=2, help="number of folds")
    parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.2)
    parser.add_argument('--use_center_of_mass', action='store_true', help='use center of mass direction', default=False)
    parser.add_argument('--use_random_dir', action='store_true', help='use random direction', default=False)
    parser.add_argument('--device', type=int, default=0, help='device')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--train_dataset', type=str, default='toxic', help='Dataset used for training')
    parser.add_argument('--api', default="",type=str, required=False)
    
    parser.add_argument('--use_ot_intervention', action='store_true', help='use ot intervention', default=False)
    parser.add_argument('--alpha_ot', type=float, default=0.1, help='alpha, intervention strength')
    parser.add_argument('--cache_dir', type=str, default="", help="hugging face hub")
    args = parser.parse_args()
    logging.set_verbosity_error()
    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # create model
    model_name_or_path = HF_NAMES[args.model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if 'gemma' in model_name_or_path.lower():
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage = True, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    
    # template of the intervened components:
    if "gpt" in args.model_name:
        template_intervened = "transformer.h.{layer}.attn.c_proj"
    else:
        template_intervened = "model.layers.{layer}.self_attn.o_proj"

    # define number of layers and heads
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    # load activations 
    head_wise_activations = np.load(f"../features/{args.train_dataset}/{args.model_name}_{args.dataset_name}_head_wise.npy")
    labels = np.load(f"../features/{args.train_dataset}/{args.model_name}_{args.dataset_name}_labels.npy")
    head_wise_activations = rearrange(head_wise_activations, 'b l (h d) -> b l h d', h = num_heads)

    # tuning dataset: no labels used, just to get std of activations along the direction
    activations_dataset = args.dataset_name if args.activations_dataset is None else args.activations_dataset
    tuning_activations = np.load(f"../features/{args.train_dataset}/{args.model_name}_{activations_dataset}_head_wise.npy")
    tuning_activations = rearrange(tuning_activations, 'b l (h d) -> b l h d', h = num_heads)
    tuning_labels = np.load(f"../features/{args.train_dataset}/{args.model_name}_{activations_dataset}_labels.npy")

    separated_head_wise_activations, separated_labels = [head_wise_activations[i:i+1] for i in range(len(head_wise_activations))], [(1 - labels[i:i+1]) for i in range(len(labels))]
    # run k-fold cross validation
    results = []
    train_idxs = np.arange(len(separated_labels))
    # pick a val set using numpy
    train_set_idxs = np.random.choice(train_idxs, size=int(len(train_idxs)*(1-args.val_ratio)), replace=False)
    val_set_idxs = np.array([x for x in train_idxs if x not in train_set_idxs])
    # get directions
    if args.use_center_of_mass:
        com_directions = get_com_directions(num_layers, num_heads, train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels)
    else:
        com_directions = None
    top_heads, probes = get_top_heads(train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels, num_layers, num_heads, args.seed, args.num_heads, args.use_random_dir)
    print("Heads intervened: ", sorted(top_heads))

    if args.use_ot_intervention:
        used_activations = torch.tensor(np.concatenate([separated_head_wise_activations[i] for i in train_set_idxs], axis = 0), dtype=torch.float32)
        y_train = 1 - np.concatenate([separated_labels[i] for i in train_set_idxs], axis = 0)
        used_labels = torch.tensor(y_train, dtype=torch.float32) 
        save_folder = f'ot_save/iti/{args.train_dataset}/{args.model_name}_seed_{args.seed}_alpha_{args.alpha_ot}_fold_{i}_top_{args.num_heads}'
        interventions = get_ot_interventions_dict(top_heads, probes, used_activations, used_labels, 0, num_heads, save_folder, alpha=args.alpha_ot, template_intervened=template_intervened)
        
        def lt_modulated_vector_add(head_output, layer_name, start_edit_location='lt'): 
            head_output = rearrange(head_output, 'b s (h d) -> b s h d', h=num_heads)
            for ijk, (head, A, b, clf, th) in enumerate(interventions[layer_name]):
                A_to_add = torch.tensor(A).to(head_output.device.index)
                b_to_add = torch.tensor(b).to(head_output.device.index)

                if start_edit_location == 'lt':
                    delta = A_to_add.half() @ head_output[:, -1, head, :].reshape(b_to_add.shape) + b_to_add.half() - head_output[:, -1, head, :].reshape(b_to_add.shape)
                    delta = delta.reshape(head_output[:, -1, head, :].shape)
                    head_output[:, -1, head, :] += args.kappa_ot * delta
                else: 
                    for loc in range(start_edit_location, head_output.shape[1]):
                        delta = A_to_add.half() @ head_output[:, loc, head, :].reshape(b_to_add.shape) + b_to_add.half() - head_output[:, loc, head, :].reshape(b_to_add.shape)
                        delta = delta.reshape(head_output[:, loc, head, :].shape)
                        head_output[:, loc, head, :] += args.kappa_ot * delta
            head_output = rearrange(head_output, 'b s h d -> b s (h d)')
            return head_output
    else:
        interventions = get_interventions_dict(top_heads, probes, tuning_activations, num_heads, args.use_center_of_mass, args.use_random_dir, com_directions, template_intervened=template_intervened)

        def lt_modulated_vector_add(head_output, layer_name, start_edit_location='lt'): 
            head_output = rearrange(head_output, 'b s (h d) -> b s h d', h=num_heads)
            for head, direction, proj_val_std in interventions[layer_name]:
                direction_to_add = torch.tensor(direction).to(head_output.device.index)
                if start_edit_location == 'lt': 
                    head_output[:, -1, head, :] += args.alpha * proj_val_std * direction_to_add
                else: 
                    head_output[:, start_edit_location:, head, :] += args.alpha * proj_val_std * direction_to_add
            head_output = rearrange(head_output, 'b s h d -> b s (h d)')
            return head_output
        
    if args.use_ot_intervention:
        filename = f'{args.model_name}_train_{args.train_dataset}_seed_{args.seed}_alpha_{int(args.alpha_ot)}'
    else:
        filename = f'{args.model_name}_train_{args.train_dataset}_seed_{args.seed}_top_{args.num_heads}_heads_alpha_{int(args.alpha)}'

    if args.use_center_of_mass:
        filename += '_com'
    if args.use_random_dir:
        filename += '_random'
    if args.use_ot_intervention:
        filename += f"_use_ot_{args.alpha_ot}"

    test_df = pd.read_json("../Toxic/nontoxic_prompts-10k.jsonl", lines=True).drop(columns=['filename', 'continuation', 'begin', 'end', 'challenging']).head(1000)
    output_path = f'results_dump/real_toxic/iti/answer_dump/test/{filename}.jsonl'
    summary_path = f'results_dump/real_toxic/iti/summary_dump/test/{filename}.jsonl'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    
    results = alt_completion_evaluate(
        models={args.model_name: model},
        metric_names=['perspective', 'fluency_and_diversity', 'ce', 'kl'],
        test_df=test_df,
        output_path=output_path,
        summary_path=summary_path,
        device="cuda", 
        interventions=interventions, 
        intervention_fn=lt_modulated_vector_add, 
        api=args.api,
    )

       
    print(results)

if __name__ == "__main__":
    main()
