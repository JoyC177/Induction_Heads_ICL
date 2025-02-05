"""
This script contains functionality for calculating and plotting prefix scores for models based on the Llama architecture.
Adapted from https://github.com/amazon-science/llm-interpret/blob/main/lm_eval/prefix_matching_copying.py
"""

import os
import torch
import pickle
import argparse
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from style import *
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import json
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--frequent_exclude_ratio", type=float, default = 0.04)
parser.add_argument("--pretrained", type = str, default = 'facebook/opt-125m')
parser.add_argument("--model_cache_dir", type = str, default = None)
parser.add_argument("--tokenizer_cache_dir", type = str, default = None)
parser.add_argument("--num_seeds", type = int, default = 5)
parser.add_argument("--use_save_outputs", action = 'store_true')

args = parser.parse_args()

# Load and modify the configuration
config = AutoConfig.from_pretrained(args.pretrained, output_attentions=True, cache_dir = args.model_cache_dir, trust_remote_code=True)

if not args.use_save_outputs:
    device_map = 'auto'
    model = AutoModelForCausalLM.from_pretrained(args.pretrained, cache_dir = args.model_cache_dir, device_map = device_map, config=config, trust_remote_code=True)
    print("Model loaded")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained,trust_remote_code=True)
    print("Tokenizer loaded")

    ## Create a ranking of BPE tokens for LLaMA models.
    ## Unlike GPT-style models that have `tokenizer.bpe_ranks`, LLaMA uses a different tokenizer structure.
    ## Instead, we extract token merges from `tokenizer.json`, which contains the merge history.
    ## More details about merging at https://huggingface.co/docs/transformers/tokenizer_summary.

    # Load the tokenizer.json file to extract token merges
    tokenizer.save_pretrained("./tokenizer_files")
    tokenizer_json_path = './tokenizer_files/tokenizer.json'

    with open(tokenizer_json_path, 'r', encoding='utf-8') as file:
        tokenizer_data = json.load(file)

    # Access the BPE merge list from the tokenizer data
    merges = tokenizer_data['model']['merges']

    # Remove temporary tokenizer files
    shutil.rmtree("./tokenizer_files")
    
    # Initialize ranked structures
    ranked_dict = dict()
    ranked_list = []
    
    print("Creating ranked dictionary")
    for merge_string in tqdm(merges):
        ## LLaMA token merges are stored as space-separated strings; remove spaces to get the token
        bpe_token = merge_string.replace(' ', '')
        if bpe_token not in ranked_list:
            ranked_list.append(bpe_token)

    # Assign ranks to BPE tokens based on merge order   
    for rank, bpe_token in enumerate(ranked_list):
        ranked_dict[rank] = tokenizer.convert_tokens_to_ids(bpe_token)

    ranked_vocab_size = len(ranked_list)
    # Exclude a fraction of frequent and rare BPE tokens from random sequences
    frequent_excluded_ranks = int(args.frequent_exclude_ratio * ranked_vocab_size)
    print(f"Excluding {frequent_excluded_ranks} most frequent and {frequent_excluded_ranks} least frequent tokens")
    rank_start, rank_end = frequent_excluded_ranks, ranked_vocab_size - frequent_excluded_ranks
    assert rank_start < rank_end and rank_end > 0
    rank_choice_list = np.arange(rank_start, rank_end)
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    final = []

    with torch.no_grad():
        print("Generating sequences")
        for seed in tqdm(range(args.num_seeds)):
            torch.manual_seed(seed)
            length = 50
            # Choose a random sequence excluding most frequent and least frequent bpe tokens
            # Generate tokens without replacement to ensure all chosen tokens are unique
            # Uniqueness ensures prefix matching scores only capture explicit repeats
            generate_ranks = np.random.choice(rank_choice_list, size=length, replace=False)

            # Append a bos_token in the beginning to ensure normal model behaviour
            generate_ids = torch.tensor([tokenizer.bos_token_id] + [ranked_dict[rank] for rank in generate_ranks])
            generate_ids = torch.unsqueeze(generate_ids, 0)
            generate_ids = generate_ids.to(0)

            # Repeat the sequence excluding the bos token
            new_generated = torch.cat([generate_ids, generate_ids[:,1:].repeat(3, 1).view(-1).unsqueeze(0)], dim = -1)
            new_generated = new_generated.to(0)
            assert new_generated.shape[1] == 4*length + 1
            out = model(input_ids = new_generated)
            attentions = out.attentions
            attn_matrix = torch.zeros((num_layers, num_heads))
            for layer in range(num_layers):
                layer_attention = attentions[layer].squeeze(0) # Remove batch dimension
                for head in range(num_heads):
                    attn_prob = layer_attention[head]
                    count = 0
                    for i in range(length + 1, 4 * length + 1):
                        token_pos_in_repeat = i % length

                        for repeat_num in range(i // length):
                            pos_in_prev_repeat = repeat_num * length + token_pos_in_repeat
                            attn_matrix[layer][head] += attn_prob[i][pos_in_prev_repeat + 1].item()
                        count += 1
    
                    attn_matrix[layer][head] = attn_matrix[layer][head] / count
        
            final.append(attn_matrix.unsqueeze(0))

    final = torch.cat(final, dim = 0)
    mean = final.mean(dim = 0)
    variance = final.var(dim = 0)
    
    # Save prefix score data
    base_dir = os.path.join("prefix_scores", args.pretrained)
    os.makedirs(base_dir, exist_ok=True)
    save_plot_path_mean = os.path.join(base_dir, "pfx_matching_mean.png")
    save_plot_path_var = os.path.join(base_dir, "pfx_matching_var.png")
    save_outputs_path = os.path.join(base_dir, "pfx_matching.pkl")

    with open(save_outputs_path, 'wb') as f:
        pickle.dump({'mean': mean, 'variance': variance}, f)

if args.use_save_outputs:
    with open(args.save_outputs, 'rb') as f:
        res = pickle.load(f)
        mean, variance = res['mean'], res['variance']
        num_layers, num_heads = mean.shape


max_, min_ = mean.max(), mean.min()
print(f"Largest prefix-matching score: {max_:.3f}")
print(f"Scores saved to {base_dir}")

# Create average prefix score heatmap
ax = sns.heatmap(mean.numpy(), xticklabels = [(i) if i%2==0 else None for i in range(num_heads)], yticklabels = [(i) if i%2==0 else None for i in range(num_layers)], vmin = 0, vmax = 1)
plt.ylabel('Layers')
plt.xlabel('Heads')
plt.title('Prefix Matching Score')
ax.invert_yaxis()

plt.savefig(save_plot_path_mean)

plt.close()

# Create prefix score variance heatmap
max_, min_ = variance.max(), variance.min()
ax = sns.heatmap(variance.numpy(), xticklabels = [(i) if i%2==0 else None for i in range(num_heads)], yticklabels = [(i) if i%2==0 else None for i in range(num_layers)], vmin = min_, vmax = max_)
plt.ylabel('Layers')
plt.xlabel('Heads')
plt.title('Prefix Matching Score')
ax.invert_yaxis()

plt.savefig(save_plot_path_var)
