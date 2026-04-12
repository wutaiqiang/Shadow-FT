"""
Weight Similarity (sigma) between a Base model and an Instruct model.

Reproduces the relative gap ratio σ from Shadow-FT (arXiv: 2505.12716, Sec 2.3):

    σ(W_A, W_B) = sum(|W_A - W_B|) / (sum(|W_A|) + sum(|W_B|))

Implementation notes:
  * Iterates over transformer layers one at a time by parsing
    ``model.safetensors.index.json`` in the Base model directory.
  * Only loads the keys belonging to the current layer, keeping memory low
    even for 70B+ checkpoints.
  * BF16 tensors are up-cast to FP32 before the subtraction.

Usage:
    python3 src/shadow/weight_similarity.py --B <base_model_dir> --I <instruct_model_dir>
"""

import os
import json
import re
import argparse

import numpy as np
import torch
from safetensors import safe_open


# -------------------------------
# Helper: parse the safetensors index to discover layers and their keys
# -------------------------------

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


# -------------------------------
# Helper: load only the requested keys from *.safetensors shards
# -------------------------------

def load_selected_weights_optimized(folder_path, selected_keys, device="cpu"):
    weights = {}
    files = [f for f in os.listdir(folder_path) if f.endswith(".safetensors")]
    requested_keys = set(selected_keys)
    found_keys = set()

    for filename in files:
        full_path = os.path.join(folder_path, filename)
        try:
            with safe_open(full_path, framework="pt", device=device) as f:
                file_keys = set(f.keys())
                intersect = requested_keys & file_keys
                for key in intersect:
                    tensor = f.get_tensor(key)
                    if tensor.dtype == torch.bfloat16:
                        tensor = tensor.float()
                    weights[key] = tensor.cpu().numpy()
                    found_keys.add(key)
        except Exception as e:
            print(f"[Error] Failed to read {full_path}: {e}")

    not_found = requested_keys - found_keys
    if not_found:
        for key in sorted(not_found):
            print(f"    [missing] {key}")

    return weights


# -------------------------------
# Metrics
# -------------------------------

def calculate_sigma(w_A, w_B):
    diff = np.abs(w_A - w_B).sum()
    total = np.abs(w_A).sum() + np.abs(w_B).sum()
    return diff / total


def calculate_sparsity_ratio(w_A, w_B, threshold=1e-5):
    return np.mean(np.abs(w_A - w_B) < threshold)


# -------------------------------
# Main
# -------------------------------

def main(base_model, instruct_model):
    folder_A = base_model
    folder_B = instruct_model
    index_json_name = "model.safetensors.index.json"

    json_path_A = os.path.join(folder_A, index_json_name)
    json_info = parse_safetensors_index(json_path_A)
    num_layers = json_info["total_layers"]
    layer_to_keys = json_info["layer_to_keys"]

    all_sigmas = []  # per-key sigma values across all layers

    print(f"Total layers: {num_layers}")

    for layer_idx in range(num_layers):
        print(f"Processing Layer {layer_idx} ...")

        keys_in_layer = layer_to_keys.get(layer_idx, [])
        if not keys_in_layer:
            continue

        A_weights = load_selected_weights_optimized(folder_A, keys_in_layer.copy(), device="cpu")
        B_weights = load_selected_weights_optimized(folder_B, keys_in_layer.copy(), device="cpu")

        sigmas = []
        for key in keys_in_layer:
            if key in A_weights and key in B_weights:
                sigma = calculate_sigma(A_weights[key], B_weights[key])
                sigmas.append(sigma)
                print(f"key: {key}, sigma: {sigma}")

        all_sigmas.extend(sigmas)

        del A_weights, B_weights
        torch.cuda.empty_cache()

    print("Average sigma across all tensors:\n", np.mean(all_sigmas))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare two LLaMA-style models' weights using the sigma metric "
                    "(Shadow-FT, arXiv: 2505.12716, Sec 2.3)."
    )
    parser.add_argument("--B", type=str, required=True,
                        help="Path to base model directory (e.g., Llama-3.1-8B/).")
    parser.add_argument("--I", type=str, required=True,
                        help="Path to instruct model directory (e.g., Llama-3.1-8B-Instruct/).")

    args = parser.parse_args()

    main(args.B, args.I)

    # Example invocations:
    # python3 src/shadow/weight_similarity.py --B Qwen3-8B-Base/ --I Qwen3-8B/
    # python3 src/shadow/weight_similarity.py --B Llama-3.1-8B/ --I Llama-3.1-8B-Instruct/
