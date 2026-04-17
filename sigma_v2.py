import os
import json
import re
import numpy as np
import torch
from safetensors import safe_open
import argparse

# -------------------------------
# 工具函数：解析 JSON，获取层数和每层 keys
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
        "layer_to_keys": layer_to_keys
    }

# -------------------------------
# 工具函数：加载指定 keys 的权重（优化版）
# -------------------------------

def load_selected_weights_optimized(folder_path, selected_keys, device="cpu"):
    weights = {}
    files = [f for f in os.listdir(folder_path) if f.endswith(".safetensors")]
    requested_keys = set(selected_keys)  # 原始请求
    found_keys = set()

    # print(f"[Loading] {len(requested_keys)} keys from {folder_path}")

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
                    # 可选：打印加载细节
                    # print(f"  Loaded {key} from {filename}")
        except Exception as e:
            print(f"[Error] Failed to read {full_path}: {e}")

    # === 关键：检查哪些 key 没找到 ===
    not_found = requested_keys - found_keys
    if not_found:
        # print(f"[❌ FAILED] These keys were NOT FOUND in {folder_path}:")
        for key in sorted(not_found):
            print(f"    {key}")
    else:
        # print(f"[✅ Success] All {len(requested_keys)} keys loaded.")
        pass

    return weights

# -------------------------------
# 工具函数：sigma 计算
# -------------------------------

def calculate_sigma(w_A, w_B):
    diff = np.abs(w_A - w_B).sum()
    total = np.abs(w_A).sum() + np.abs(w_B).sum()
    return diff / total 

def calculate_sparsity_ratio(w_A, w_B, threshold=1e-5):
    return np.mean(np.abs(w_A - w_B) < threshold)

# -------------------------------
# 主程序逻辑
# -------------------------------

def main(base_model, instruct_model):
    folder_A = base_model
    folder_B = instruct_model
    index_json_name = "model.safetensors.index.json"

    json_path_A = os.path.join(folder_A, index_json_name)
    json_info = parse_safetensors_index(json_path_A)
    num_layers = json_info["total_layers"]
    layer_to_keys = json_info["layer_to_keys"]

    all_sigmas = []  # 存储每层的平均 sigma

    print(f"Total layers: {num_layers}")

    for layer_idx in range(num_layers):
        print(f"Processing Layer {layer_idx} ...")

        keys_in_layer = layer_to_keys.get(layer_idx, [])
        if not keys_in_layer:
            continue

        # print(keys_in_layer)

        A_weights = load_selected_weights_optimized(folder_A, keys_in_layer.copy(), device="cpu")
        B_weights = load_selected_weights_optimized(folder_B, keys_in_layer.copy(), device="cpu")

        # print(A_weights.keys())

        sigmas = []
       
        for key in keys_in_layer:
            if key in A_weights and key in B_weights: # and "norm" not in key:
                sigma = calculate_sigma(A_weights[key], B_weights[key])
                sigmas.append(sigma)
                print(f"key: {key}, sigma: {sigma}")
                
        # print(f"layer {layer_idx} sigma:", np.mean(sigmas))
        # print("{:.3f}".format(np.mean(sigmas))) 


        all_sigmas.extend(sigmas)

        del A_weights, B_weights
        torch.cuda.empty_cache()

    
    # print("All sigmas (shape):", np.array(all_sigmas).shape)
    # print(base_model)
    print("Average sigma across all tensors:\n", np.mean(all_sigmas))
    # print("All: \n", all_sigmas)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two LLaMA models' weights using sigma metric.")
    parser.add_argument("--B", type=str, required=True, help="Path to base model (e.g., llama-3.1-8B)")
    parser.add_argument("--I", type=str, required=True, help="Path to instruct model (e.g., llama-3.1-8B-Instruct)")

    args = parser.parse_args()

    main(args.B, args.I)

    # python3 sigma_v2.py --B Qwen3-8B-Base/ --I Qwen3-8B/
    # python3 sigma_v2.py --B INS/ --I THINK/  0.0419
    # python3 sigma_v2.py --B BASE/ --I INS/   0.0871
    # python3 sigma_v2.py --B BASE/ --I THINK/ 0.0988

    # python3 sigma_v2.py --B MIX/ --I INS/    0.0914
    # python3 sigma_v2.py --B BASE/ --I MIX/   0.0506
    # python3 sigma_v2.py --B MIX/ --I THINK/  0.0981

    # sigma
    # python3 sigma_v2.py --B Qwen3-4B-Base/ --I Qwen3-4B-Base-Ins/          0.0558
    # python3 sigma_v2.py --B Qwen3-4B-Base/ --I Qwen3-4B-Instruct-2507/     0.0562
    # python3 sigma_v2.py --B Qwen3-4B-Base-Ins/ --I Qwen3-4B-Instruct-2507/ 0.0057
    # python3 sigma_v2.py --B Qwen3-4B-Base-Ins/ --I Qwen3-4B-Base-Ins-woEmb/

    # python3 sigma_v2.py --B Qwen3.5-35B-A3B-Base --I Qwen3.5-35B-A3B  