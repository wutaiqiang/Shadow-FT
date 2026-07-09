"""
Multi-process Weight Similarity (sigma) between a Base model and an Instruct model.

Same as weight_similarity.py but parallelizes across layers using multiprocessing.

Usage:
    python3 weight_similarity_multi.py --B <base_model_dir> --I <instruct_model_dir> --workers 12
"""

import os
import json
import re
import argparse
from multiprocessing import Pool
from functools import partial

import numpy as np
import torch
from safetensors import safe_open


def parse_safetensors_index(json_path):
    with open(json_path, "r") as f:
        index_data = json.load(f)

    weight_map = index_data["weight_map"]
    layer_pattern = re.compile(r"model\.layers\.(\d+)\.")
    layers = set()
    layer_to_keys = {}

    for key in weight_map.keys():
        match = layer_pattern.search(key)
        if match:
            layer_idx = int(match.group(1))
            layers.add(layer_idx)
            if layer_idx not in layer_to_keys:
                layer_to_keys[layer_idx] = []
            layer_to_keys[layer_idx].append(key)

    total_layers = max(layers) + 1 if layers else 0
    return {
        "total_layers": total_layers,
        "layer_to_keys": layer_to_keys,
    }


def load_selected_weights(folder_path, selected_keys):
    weights = {}
    files = [f for f in os.listdir(folder_path) if f.endswith(".safetensors")]
    requested_keys = set(selected_keys)

    for filename in files:
        full_path = os.path.join(folder_path, filename)
        try:
            with safe_open(full_path, framework="pt", device="cpu") as f:
                file_keys = set(f.keys())
                intersect = requested_keys & file_keys
                for key in intersect:
                    tensor = f.get_tensor(key)
                    if tensor.dtype == torch.bfloat16:
                        tensor = tensor.float()
                    weights[key] = tensor.numpy()
        except Exception:
            pass

    return weights


def calculate_sigma(w_A, w_B):
    diff = np.abs(w_A - w_B).sum()
    total = np.abs(w_A).sum() + np.abs(w_B).sum()
    return diff / total


def process_layer(layer_idx, folder_A, folder_B, layer_to_keys):
    keys_in_layer = layer_to_keys.get(layer_idx, [])
    if not keys_in_layer:
        return []

    A_weights = load_selected_weights(folder_A, keys_in_layer)
    B_weights = load_selected_weights(folder_B, keys_in_layer)

    results = []
    for key in keys_in_layer:
        if key in A_weights and key in B_weights:
            sigma = calculate_sigma(A_weights[key], B_weights[key])
            results.append((key, sigma))

    return results


def main(base_model, instruct_model, workers):
    index_json_name = "model.safetensors.index.json"
    json_path_A = os.path.join(base_model, index_json_name)
    json_info = parse_safetensors_index(json_path_A)
    num_layers = json_info["total_layers"]
    layer_to_keys = json_info["layer_to_keys"]

    print(f"Total layers: {num_layers}")
    print(f"Workers: {workers}")

    process_fn = partial(process_layer, folder_A=base_model, folder_B=instruct_model, layer_to_keys=layer_to_keys)

    all_results = []
    with Pool(workers) as pool:
        for layer_results in pool.imap(process_fn, range(num_layers)):
            for key, sigma in layer_results:
                all_results.append((key, sigma))
                print(f"  {key}: sigma={sigma:.6f}")

    all_sigmas = [s for _, s in all_results]
    print(f"\n{'='*60}")
    print(f"Total keys compared: {len(all_sigmas)}")
    print(f"Average sigma: {np.mean(all_sigmas):.6f}")
    print(f"Median sigma:  {np.median(all_sigmas):.6f}")
    print(f"Max sigma:     {np.max(all_sigmas):.6f}")
    print(f"Min sigma:     {np.min(all_sigmas):.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=str, required=True, help="Path to base model directory.")
    parser.add_argument("--I", type=str, required=True, help="Path to instruct/finetuned model directory.")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel workers.")
    args = parser.parse_args()

    main(args.B, args.I, args.workers)
