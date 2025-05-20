# Shadow-FT
Official code for paper "Shadow-FT: Tuning Instruct via Base"

  <a href="https://arxiv.org/pdf/2505.12716"><b>[📜 Paper]</b></a> •
  <a href="TBD"><b>[🤗 HF Models]</b></a> •
  <a href="https://github.com/wutaiqiang/Shadow-FT"><b>[🐱 GitHub]</b></a>

This repo contains the code for our paper: <a href="https://arxiv.org/abs/2411.06839" target="_blank">Shadow-FT: Tuning Instruct via Base</a> by <a href="https://wutaiqiang.github.io" target="_blank">Taiqiang Wu</a> <a href="https://rummyyang.github.io/" target="_blank">Runming Yang</a>, Jiayi Li, Pengfei Hu, Ngai Wong and Yujiu Yang.

There is an <a href="TBD" target="_blank"> explanation blog </a> for this paper (in Chinese).


****************TBD**************


## Overview

<img src="https://github.com/user-attachments/assets/277dcdf4-c599-41be-97f6-f56a678b4865" width="50%" />


## Quick start

Our code is basically built on  <a href="https://github.com/hiyouga/LLaMA-Factory" target="_blank">LLaMA-Factory</a>, which is a tremendous project and you can find everything you wonder there, thanks to their great framework and nice code!


### Environment:
```
git clone --depth 1 https://github.com/yang3121099/LLM-Neo.git
cd LLM-Neo
pip install -e ".[torch,metrics]"
```

### Basic usage:
```
python3 script_Neo.py
bash run_train.sh
```
The `script_Neo.py` can generate all `4 training-strategy yaml` once time, you can use `--run` to choose from `[SFT, LoRA, KD, Neo]`, the settings will be generated to `examples/train_neo` and you can modify them manually.


### Advanced usage:
```
# for basic hypermeters
python3 script_Neo.py --base_lr 1e-5 --epochs 3 --batch_size 16 --grad_accum 4 --max_samples 1000 

# for LoRA and Neo
python3 script_Neo.py --run lora neo --lora_rank 32 --base_model meta-llama/Meta-Llama-3-8B-Instruct

# for KD and Neo
python3 script_Neo.py --run kd neo --base_model meta-llama/Meta-Llama-3-8B-Instruct --teacher_model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B 
```

#### Learning Rate
Follow the guideline propused by LLM-Neo paper, we set `lr' = 10 * lr` for LoRA and Neo automatically, please pay attention and it can be changed in your way.

#### Knowledge Distillation
LLM-Neo is the combination of LoRA and KD, while KD is not originally supported by LLaMA-Factory.

We add the `--teacher_model` and `--kd_ratio` parameters in `src/llamafactory/hparams/finetuning_args` and `src/llamafactory/train/sft/trainer`, which is easy to extend to other methods when you DIY.


## ☕️ Citation

If you find this repository helpful, please consider citing our paper:

```
@article{yang2024llm,
  title={Llm-neo: Parameter efficient knowledge distillation for large language models},
  author={Yang, Runming and Wu, Taiqiang and Wang, Jiahao and Hu, Pengfei and Wong, Ngai and Yang, Yujiu},
  journal={arXiv preprint arXiv:2411.06839},
  year={2024}
}
```
