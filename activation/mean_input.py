"""
This script contains functionality for calculating and storing mean activations over a reference distribution (MNLI Premises).
"""

import pyvene as pv
import torch
import copy
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import random
import os
import argparse
from collections import defaultdict
import pickle
import math
import shutil

# Define storage directories
STORAGE_DIR = "projects/" # For saving mean activations
TMP_DIR = "scratch-local/" # For saving individual activations (will be deleted after mean activations are saved)

def load_model(model_name, cache_dir):
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir,use_fast=True)

    return model, tokenizer

def average_tensors(directory, subdir=None):
    """Computes the average of tensors stored in a directory and saves the mean tensor.
    
    Args:
        directory (str): The base directory where tensors are stored.
        subdir (str, optional): The specific subdirectory containing tensors.
    """
    files = os.listdir(f"{directory}/{subdir}")

    mean_tensor = None
    for file in files:

        tensor_path = os.path.join(f"{directory}/{subdir}", file)

        tensor = torch.load(tensor_path)

        if mean_tensor is None:
            mean_tensor = tensor
        else:
            mean_tensor += tensor
    
    mean_tensor /= len(files)
    os.makedirs(f"{STORAGE_DIR}{args.pretrained}/{subdir}", exist_ok=True)
    torch.save(mean_tensor, f"{STORAGE_DIR}{args.pretrained}/{subdir}/mean.pt")

def find_top_scoring_heads(prefix_scores, num_heads, num_heads_per_layer, printing=True):
    """Finds the top-scoring attention heads based on prefix scores.
    
    Args:
        prefix_scores (Tensor): Tensor containing prefix scores for each head.
        num_heads (int): Number of top heads to find.
        num_heads_per_layer (int): Number of heads per layer in the model.
        printing (bool, optional): Whether to print the top scores.
    
    Returns:
        dict: Dictionary with layer indices as keys and lists of head indices to ablate as values.
    """
    # Flatten the tensor
    flat_tensor = prefix_scores.view(-1)

    # Find the top N scores and their indices in the flattened tensor
    top_values, flat_indices = torch.topk(flat_tensor, num_heads)

    # Convert flat indices to 2D indices
    rows = flat_indices // num_heads_per_layer
    cols = flat_indices % num_heads_per_layer

    # Display the results
    if printing:
        print(f"Top {num_heads} prefix-matching scores: ")
        for i in range(num_heads):
            print(f"Score: {top_values[i]}, Location: ({rows[i]}, {cols[i]})")

    # Create the ablation dictionary
    ablation_dict = defaultdict(list)
    for (row, col) in zip(rows, cols):
        ablation_dict[row.item()].append(col.item())

    return ablation_dict

def pv_saving(cached_w, layer_name, b, s):
    """Caches the activations for a given layer."""
    cached_w[f"{layer_name}"] = copy.deepcopy(b.data)


def create_intervention_config(cached_w, layers, model_type):
    """Creates the intervention configuration for the model.
    
    Args:
        layers (list): List of layers to save activations for.
        model_type (str): Type of model (internlm or llama).
        
    Returns:
        list: List of dictionaries containing the intervention configuration.
    """
    pv_config = []
    for layer in layers:
        # Skip if the mean activations have already been saved
        if os.path.exists(f"{STORAGE_DIR}{args.pretrained}/{layer}/mean.pt"):
            continue

        # Define the component to save activations for
        component = f"model.layers[{layer}].attention.input" if model_type == 'internlm' else f"model.layers[{layer}].self_attn.input"

        # Define the intervention function
        pv_config.append({
            "component": component,
            "intervention": lambda b, s, layer_name=layer: pv_saving(cached_w, layer_name, b, s)
        })
    return pv_config

def main(args):
    ds = load_dataset("glue", "mnli")

    premises = ds['train']['premise']
    iterate = iter(premises)
    
    model, tokenizer = load_model(args.pretrained, args.model_cache_dir)

    # Load prefix scores
    with open(f'../induction_scores/prefix_scores/{args.pretrained}/pfx_matching.pkl', 'rb') as file:
        data = pickle.load(file)
    prefix_scores = data['mean']

    # Calculate maximum number of heads to ablate
    n_ablate = math.ceil(model.config.num_attention_heads * model.config.num_hidden_layers * 0.03)

    # Find top-scoring heads
    ablation_dict_ind = find_top_scoring_heads(prefix_scores, n_ablate, model.config.num_attention_heads, printing=False)

    new_dataset = []

    # Create 500 sequences of length 3300
    for _ in range(500):

        counter = 0
        sentence = ""

        while counter < 3300:
            next_premise = next(iterate)
            tokenized = tokenizer.encode(next_premise)
            length = len(tokenized)
            counter += length
            sentence += next_premise + " "

        new_dataset.append(sentence)
    
    # Define the layers to save activations for
    layers_to_save = list(ablation_dict_ind.keys())

    cached_w = {}
    model_type = "internlm" if "internlm" in args.pretrained else "llama"
    intervention_config = create_intervention_config(cached_w, layers_to_save, model_type)

    # Define the intervention model
    pv_internlm = pv.IntervenableModel(intervention_config, model=model)

    # Iterate through the newly created dataset and save activations for each sample
    for sentence_num, sentence in enumerate(new_dataset):

        base = tokenizer(sentence, return_tensors="pt", add_special_tokens=False)
        base['input_ids'] = base['input_ids'][:,:3000]
        base['attention_mask'] = base['attention_mask'][:,:3000]

        ablated_attn_w = pv_internlm(
            base = base#, unit_locations={"base": [0]}
        )

        # Save activations for each layer
        for layer, activation in cached_w.items():
            os.makedirs(f"{TMP_DIR}{args.pretrained}/{layer}", exist_ok=True)
            torch.save(activation, f"{TMP_DIR}{args.pretrained}/{layer}/{sentence_num}.pt")

    # Average the activations for each layer
    for layer in cached_w.keys():
        average_tensors(f"{TMP_DIR}{args.pretrained}", f"{layer}")

    # Remove temporary directory
    shutil.rmtree(TMP_DIR)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--pretrained", type=str, default=None)
    argparser.add_argument("--model_cache_dir", type = str, default = None)
    args = argparser.parse_args()

    main(args)

