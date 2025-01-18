import os
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, logging

HF_NAMES = {
    'qwen_2.5_1.5B': 'Qwen/Qwen2.5-1.5B',
    'qwen_2.5_1.5B-math': 'Qwen/Qwen2.5-Math-1.5B',
}

def get_qwen_activations(model, input_ids, device):
    """Extract hidden states from Qwen model."""
    with torch.no_grad():
        outputs = model(input_ids.to(device), output_hidden_states=True)
        hidden_states = outputs.hidden_states
        # get only last token hidden states 
        all_hidden_states = torch.stack(hidden_states)[:, :, -1, :]
        return all_hidden_states.detach().cpu().to(dtype=torch.float16).numpy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='qwen_2.5_1.5B',
                        choices=['qwen_2.5_1.5B', 'qwen_2.5_1.5B-math'])
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--cache_dir', type=str, default=None)
    args = parser.parse_args()

    logging.set_verbosity_error()
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    
    model_name_or_path = HF_NAMES[args.model_name]
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
    )

    # qwen use bfloat instead of float, so if we force torch.float16, it will raise Nan output...
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map=device,
        torch_dtype='auto',
        low_cpu_mem_usage=True,
    )
    model.eval()

    dataset = load_dataset('json', data_files='../data/prm800k_test.jsonl')['train']
    selected_keys = ['problem', 'solution', 'answer']
    
    all_hidden_states = []
    print("Extracting hidden states...")
    for item in tqdm(dataset):
        text = '\n '.join(map(item.get, selected_keys))
        inputs = tokenizer(text, return_tensors='pt')
        hidden_states = get_qwen_activations(model, inputs.input_ids, device)
        all_hidden_states.append(hidden_states)

    os.makedirs('../features/prm800k_test', exist_ok=True)
    output_path = f'../features/prm800k_test/{args.model_name}_hidden_states.npy'
    # breakpoint()
    np.save(output_path, np.array(all_hidden_states))
    print(f"Hidden states saved to {output_path}")

if __name__ == '__main__':
    main()
