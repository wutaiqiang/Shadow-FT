import os
import shutil
import argparse
import torch
from safetensors import safe_open
from safetensors.torch import save_file
import re
import json

# NEW: use the official helper from huggingface_hub that supersedes
# transformers.modeling_utils.shard_checkpoint
from huggingface_hub import save_torch_state_dict

# Optional accelerate-based loading removed; use manual shard loading instead

def is_linear_param(name):
    """Return True if the parameter belongs to a linear/projection layer."""
    patterns = [
        r"k_proj", r"q_proj", r"v_proj",
        r"o_proj", r"up_proj", r"gate_proj", r"down_proj"
    ]
    return any(re.search(pattern, name.lower()) for pattern in patterns)


def load_weights(model_path):
    """Load weights from either *.safetensors shards, a single pytorch_model.bin, or manual .bin shards."""
    weights = {}

    # All .safetensors shards in the directory (ignore the index file)
    safetensor_files = [
        os.path.join(model_path, f)
        for f in os.listdir(model_path)
        if f.endswith(".safetensors") and not f.endswith(".safetensors.index.json")
    ]

    pytorch_bin = os.path.join(model_path, "pytorch_model.bin")
    index_json = os.path.join(model_path, "pytorch_model.bin.index.json")

    if safetensor_files:
        for model_file_path in safetensor_files:
            with safe_open(model_file_path, framework="pt", device="cpu") as f:
                for k in f.keys():
                    weights[k] = f.get_tensor(k)
    # single-file .bin
    elif os.path.exists(pytorch_bin):
        print(f"Loading PyTorch model from {pytorch_bin}")
        state_dict = torch.load(pytorch_bin, map_location="cpu")
        weights = state_dict
    # manual .bin shards
    elif os.path.exists(index_json):
        print(f"Loading shard bins listed in index {index_json}")
        # gather all shard files
        shard_files = sorted(
            os.path.join(model_path, f)
            for f in os.listdir(model_path)
            if re.match(r"pytorch_model-\d{5}-of-\d{5}\.bin", f)
        )
        for shard_path in shard_files:
            print(f"Loading shard {shard_path}")
            shard_dict = torch.load(shard_path, map_location="cpu")
            weights.update(shard_dict)
    else:
        raise FileNotFoundError(
            f"No .safetensors or pytorch_model.bin found in {model_path}")

    return weights


def save_safetensor_weights(model_path, weights, max_shard_size="5GB"):
    """Save *weights* to *model_path* as (optionally sharded) SafeTensors files.

    Uses `huggingface_hub.save_torch_state_dict`, which handles:
      * shared tensors (e.g. tied embeddings ↔︎ lm_head)
      * automatic sharding with an accompanying index file
      * both pickle and safetensors formats (we enforce safetensors here)
    """
    os.makedirs(model_path, exist_ok=True)

    save_torch_state_dict(
        state_dict=weights,
        save_directory=model_path,
        max_shard_size=max_shard_size,
        filename_pattern="model{suffix}.safetensors",
        safe_serialization=True,
    )


def copy_tokenizer_and_config(src_dir, dst_dir):
    """Copy tokenizer/config/README/instruct .py etc. so that the merged checkpoint is fully usable."""
    for filename in os.listdir(src_dir):
        if filename.endswith(".safetensors.index.json"):
            continue
        if filename.startswith((
                "config", "tokenizer", "special", "generation")) \
           or filename.endswith(('.md', '.json', '.py')):
            src_file = os.path.join(src_dir, filename)
            dst_file = os.path.join(dst_dir, filename)
            if os.path.isfile(src_file):
                shutil.copy2(src_file, dst_file)


def print_debug_info(a_weights, b_weights, c_weights, new_weights):
    """Print shapes and simple statistics for q_proj of the first layer."""
    q_proj_weight_name = None

    for k in a_weights.keys():
        if re.search(r"layers\.0.*self_attn\.q_proj\.weight", k):
            q_proj_weight_name = k
            break

    if not q_proj_weight_name:
        print("[DEBUG] Could not find q_proj weight name – showing first 10 keys:")
        for i, k in enumerate(list(a_weights.keys())[:10]):
            print(f"  {i}: {k}")
        return

    print("===== DEBUG: first layer self_attn.q_proj =====")
    for tag, w in ("A", a_weights), ("B", b_weights), ("C", c_weights), ("NEW", new_weights):
        tensor = w.get(q_proj_weight_name)
        print(f"{tag} weight shape: {getattr(tensor, 'shape', 'N/A')}")
    if q_proj_weight_name in c_weights:
        delta_weight = a_weights[q_proj_weight_name] - c_weights[q_proj_weight_name]
        print("Delta weight mean (a-c):", delta_weight.mean().item())
        print("Abs delta weight mean (|a-c|):", delta_weight.abs().mean().item())
    print("==============================================")


def process_single_model(tuned_model, target_model, base_model):
    print(f"\n[+] Processing tuned model: {tuned_model}")

    a_weights = load_weights(tuned_model)
    b_weights = load_weights(target_model)
    c_weights = load_weights(base_model)

    delta_weights = {k: a_weights[k] - c_weights[k]
                     for k in a_weights if k in c_weights and is_linear_param(k)}

    new_weights = {k: (b_weights[k] + delta_weights[k]) if k in delta_weights else v
                   for k, v in b_weights.items()}

    delta_dir = os.path.join(tuned_model, "merged-B2I")
    os.makedirs(delta_dir, exist_ok=True)

    print_debug_info(a_weights, b_weights, c_weights, new_weights)

    print(f"[+] Saving merged weights to {delta_dir} (sharded safetensors…)")
    save_safetensor_weights(delta_dir, new_weights)
    copy_tokenizer_and_config(target_model, delta_dir)
    print(f"[✓] Finished processing {tuned_model} → {delta_dir}")


def find_model_paths(parent_dir):
    files = os.listdir(parent_dir)
    if any(f.endswith(".safetensors") for f in files) or "pytorch_model.bin" in files:
        return [parent_dir]
    model_paths = []
    for root, _, files in os.walk(parent_dir):
        if any(f.endswith(".safetensors") for f in files) or "pytorch_model.bin" in files:
            model_paths.append(root)
    return model_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply delta = (a - c) + b for linear layers and save shards in SafeTensors format.")
    parser.add_argument("--tuned_model", type=str, required=True,
                        help="Parent dir containing fine-tuned (a) model(s) or a single a-model dir.")
    parser.add_argument("--target_model", type=str, required=True,
                        help="Path to the Instruct (b) model directory.")
    parser.add_argument("--base_model", type=str, required=True,
                        help="Path to the Base (c) model directory.")
    parser.add_argument("--max_shard_size", type=str, default="2GB",
                        help="Max size per shard, e.g. 2GB, 1_500_000_000. Default 2GB.")

    args = parser.parse_args()

    model_paths = find_model_paths(args.tuned_model)
    print(f"Found {len(model_paths)} model path(s): {model_paths}")

    for model_path in model_paths:
        process_single_model(
            tuned_model=model_path,
            target_model=args.target_model,
            base_model=args.base_model,
        )
