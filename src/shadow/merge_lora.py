#!/usr/bin/env python3
# merge_lora.py

import argparse
import os
import subprocess
import yaml
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", required=True,
                        help="Path to the LoRA finetune output directory (no checkpoint-*).")
    parser.add_argument("--target_base", required=True,
                        help="Path to the base model to which we want to merge.")
    parser.add_argument("--merge_tag", required=True,
                        help="Tag of merged directory, e.g. B2I, I2I, etc.")
    parser.add_argument("--template", default="llama3", help="Template used in training.")
    args = parser.parse_args()

    # base config
    merge_config = {
        "model_name_or_path": args.target_base,
        "adapter_name_or_path": args.adapter_path,
        "template": args.template,
        "finetuning_type": "lora",
        "export_size": 5,
        "export_device": "cpu",
        "export_legacy_format": False,
        "trust_remote_code": True
    }
    adapter_dir = Path(args.adapter_path)
    merged_dir = adapter_dir / f"merged-{args.merge_tag}"

    if merged_dir.exists():
        if any(f.suffix == ".safetensors" for f in merged_dir.iterdir()):
            print(f"[skip] {merged_dir} alreadly has safetensors files and pass.")
            return

    merged_dir.mkdir(parents=True, exist_ok=True)

    merge_config["export_dir"] = str(merged_dir)

    config_file = merged_dir / "merge_lora_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(merge_config, f)

    # run llamafactory-cli export
    cmd = ["llamafactory-cli", "export", str(config_file)]
    print(f"[merging] adapter={args.adapter_path} => base={args.target_base}, output to {merged_dir}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
