#!/usr/bin/env bash
###############################################################################
##### 0. Globals                                                             #####
###############################################################################
WORKSPACE_DIR="$(cd "$(dirname "$0")" && pwd)"
# WORKSPACE_DIR="."
RESULTS_DIR="$WORKSPACE_DIR/results"
SCRIPT_OUTPUT_DIR="$WORKSPACE_DIR/scripts"
mkdir -p "$RESULTS_DIR" "$SCRIPT_OUTPUT_DIR"


# --- LoRA switch -------------------------------------------------------------
USE_LORA=true                   # false -> full SFT
is_lora()   { [[ "${USE_LORA,,}" == "true" ]]; }

lora_ranks=(8)
ratios=(0.5)
learning_rates_lora=(2e-4)
learning_rates_sft=(1e-5)

if is_lora; then
  learning_rates=$learning_rates_lora
else
  learning_rates=$learning_rates_sft
fi

MODEL_DIR="" # ""→HF
BASE_MODELS=(
  # "Qwen2.5-14B"   # extend as needed
  "Qwen3-0.6B-Base"
  )   


# --- Helpers -----------------------------------------------------------------
sci2dec()   { printf "%.15f" "$1" | sed -E 's/0+$//;s/\.$/.0/;s/\.?0+$//'; }
maybe_rel() { [[ $1 = /* ]] && echo "$1" || echo "../$1"; }
add_eval()  { EVAL_LINES+=("('$1','$2'),"); }

###############################################################################
##### 1. Base-model resolution (no jq)                                       #####
###############################################################################

MODEL_PAIR_FILE="$WORKSPACE_DIR/examples/model_pair.json"
MODEL_PAIRS=()                  # will contain "<base>||<instruct>"

for NAME in "${BASE_MODELS[@]}"; do
  # --- extract object block --------------------------------------------------
  BLOCK=$(awk -v n="\"$NAME\"" '
    $0~n {print; getline;
           while ($0 !~ /\}/) {print; getline}; print; exit}' \
    "$MODEL_PAIR_FILE")
  [[ -z $BLOCK ]] && { echo "ERROR: '$NAME' not found in JSON"; exit 1; }

  HF_BASE=$(printf '%s\n' "$BLOCK" | sed -n 's/.*"hf_base_path":[[:space:]]*"\([^"]*\)".*/\1/p')
  HF_INST=$(printf '%s\n' "$BLOCK" | sed -n 's/.*"hf_instruct_path":[[:space:]]*"\([^"]*\)".*/\1/p')

  [[ -z $HF_BASE || -z $HF_INST ]] && { echo "ERROR: malformed JSON for $NAME"; exit 1; }

  if [[ -n $MODEL_DIR ]]; then
      REL_BASE=${HF_BASE#*/}
      REL_INST=${HF_INST#*/}
      BASE_PATH="$MODEL_DIR/$REL_BASE"
      INST_PATH="$MODEL_DIR/$REL_INST"
      [[ -d $BASE_PATH ]] || { echo "ERROR: $BASE_PATH missing"; exit 1; }
      [[ -d $INST_PATH ]] || { echo "ERROR: $INST_PATH missing"; exit 1; }
  else
      BASE_PATH=$HF_BASE
      INST_PATH=$HF_INST
  fi

  MODEL_PAIRS+=("${BASE_PATH}||${INST_PATH}")
  echo "INFO  Base=$BASE_PATH  Instruct=$INST_PATH"
done

###############################################################################
##### 2. Generate train scripts                                              #####
###############################################################################
MONTHDAY=$(date +%m%d)
TIMESTAMP=$(date +%m%d%H%M%S)

for PAIR in "${MODEL_PAIRS[@]}"; do
  B_MODEL="${PAIR%%||*}"
  I_MODEL="${PAIR##*||}"
  MODEL_BASE=$(basename "$B_MODEL")
  SCRIPT_FILE="$SCRIPT_OUTPUT_DIR/train_${MODEL_BASE}_${TIMESTAMP}.sh"
  : > "$SCRIPT_FILE"            # truncate / create

  # ================= header =================
  echo "##### Auto-generated $(date '+%F %T') #####"        >> "$SCRIPT_FILE"
  echo "# Model     : $MODEL_BASE"                         >> "$SCRIPT_FILE"
  echo "# LoRA mode : $USE_LORA"                           >> "$SCRIPT_FILE"

  # ================= template =================
  case "$B_MODEL" in
    *Llama-3*)  template="llama3" ;;
    *Qwen2*)    template="qwen" ;;
    *Llama-2*)  template="llama2" ;;
    *Qwen3*)    template="qwen3" ;;
    *internlm2*)template="intern2" ;;
    *mistral_small*) template="mistral_small" ;;
    *Mistral*)  template="mistral" ;;
    *Falcon*)   template="falcon" ;;
    *gemma3*)   template="gemma3" ;;
    *gemma*)    template="gemma" ;;
    *Yi*)       template="yi" ;;
    *Baichuan*) template="baichuan2" ;;
    *) echo "ERROR: unknown template for $B_MODEL"; exit 1 ;;
  esac
  echo "# Template  : $template"                           >> "$SCRIPT_FILE"
  echo ""                                                  >> "$SCRIPT_FILE"

  # ================= env block =================
  echo "##### Environment #####"                          >> "$SCRIPT_FILE"
  echo "export VLLM_WORKER_MULTIPROC_METHOD=spawn"        >> "$SCRIPT_FILE"
  echo "export HF_HUB_OFFLINE=0"                          >> "$SCRIPT_FILE"
  echo "export HF_DATASETS_OFFLINE=0"                     >> "$SCRIPT_FILE"
  echo "export HF_DATASETS_TRUST_REMOTE_CODE=1"           >> "$SCRIPT_FILE"
  echo "export TRUST_REMOTE_CODE=True"                    >> "$SCRIPT_FILE"
  echo "export HF_ALLOW_CODE_EVAL=1"                      >> "$SCRIPT_FILE"
  echo ""                                                 >> "$SCRIPT_FILE"

  # ================= constants =================
  DEEPSPEED_CFG="$WORKSPACE_DIR/examples/deepspeed/ds_z3_config.json"
  DATASET="Shadow_2k"
  suffix_name="Shadow_2k"
  cutoff_len=4096
  samples=(2000)
  logging_steps=1; save_steps=1000; per_device_train_batch_size=2
  gradient_accumulation_steps=16; num_train_epochs=1
  lr_scheduler_type="cosine"; warmup_ratio=0.1; bf16=true
  val_size=0.01; per_device_eval_batch_size=1
  eval_strategy="steps"; eval_steps=10000; overwrite_cache=false

  # ================= training =================
  echo "##### Training #####"                            >> "$SCRIPT_FILE"

  EVAL_LINES=()   # reset per base model

  generate_train() {
    local M_PATH=$1 TAG=$2 LR=$3
    local DEC=$(sci2dec "$LR")
    local LR_TAG="lr${DEC}"
    local OUT_ROOT="$RESULTS_DIR/${MONTHDAY}/result-${MODEL_BASE}-${MONTHDAY}"

    for MAX in "${samples[@]}"; do
      local K="$((MAX/1000))k"
      local DIR="${TAG}-${K}-$(is_lora && echo lora-rank${lora_ranks[0]} || echo sft)-${LR_TAG}-${suffix_name}"
      local OUTDIR="$OUT_ROOT/$DIR"

      echo "###### ${TAG}  max=$MAX  lr=$LR ######"         >> "$SCRIPT_FILE"
      echo "mkdir -p \"$OUTDIR\""                         >> "$SCRIPT_FILE"
      echo "cd \"$WORKSPACE_DIR\""                        >> "$SCRIPT_FILE"
      echo "llamafactory-cli train \\"                    >> "$SCRIPT_FILE"
      echo "  --model_name_or_path \"$M_PATH\" \\" >> "$SCRIPT_FILE"
      echo "  --stage sft \\"                             >> "$SCRIPT_FILE"
      echo "  --do_train true \\"                         >> "$SCRIPT_FILE"
      if is_lora; then
        echo "  --finetuning_type lora --lora_rank ${lora_ranks[0]} \\" >> "$SCRIPT_FILE"
      else
        echo "  --finetuning_type full \\"                >> "$SCRIPT_FILE"
      fi
      echo "  --deepspeed examples/deepspeed/ds_z3_config.json \\"          >> "$SCRIPT_FILE"
      echo "  --dataset \"$DATASET\" \\"                  >> "$SCRIPT_FILE"
      echo "  --template \"$template\" \\"                >> "$SCRIPT_FILE"
      echo "  --cutoff_len $cutoff_len \\"                >> "$SCRIPT_FILE"
      echo "  --max_samples $MAX \\"                      >> "$SCRIPT_FILE"
      echo "  --output_dir \"$OUTDIR\" \\" >> "$SCRIPT_FILE"
      echo "  --per_device_train_batch_size $per_device_train_batch_size \\" >> "$SCRIPT_FILE"
      echo "  --gradient_accumulation_steps $gradient_accumulation_steps \\" >> "$SCRIPT_FILE"
      echo "  --learning_rate $LR \\"                     >> "$SCRIPT_FILE"
      echo "  --num_train_epochs $num_train_epochs \\"    >> "$SCRIPT_FILE"
      echo "  --logging_steps $logging_steps \\"          >> "$SCRIPT_FILE"
      echo "  --save_steps $save_steps \\"                >> "$SCRIPT_FILE"
      echo "  --plot_loss true \\"                        >> "$SCRIPT_FILE"
      echo "  --lr_scheduler_type $lr_scheduler_type \\"  >> "$SCRIPT_FILE"
      echo "  --warmup_ratio $warmup_ratio \\"            >> "$SCRIPT_FILE"
      echo "  --bf16 $bf16 \\"                            >> "$SCRIPT_FILE"
      echo "  --val_size $val_size \\"                    >> "$SCRIPT_FILE"
      echo "  --per_device_eval_batch_size $per_device_eval_batch_size \\" >> "$SCRIPT_FILE"
      echo "  --eval_strategy $eval_strategy \\"          >> "$SCRIPT_FILE"
      echo "  --eval_steps $eval_steps \\"                >> "$SCRIPT_FILE"
      echo "  --trust_remote_code True \\"                >> "$SCRIPT_FILE"
      echo "  --flash_attn fa2 \\"                        >> "$SCRIPT_FILE"
      echo "  --overwrite_cache $overwrite_cache"         >> "$SCRIPT_FILE"
      echo ""                                             >> "$SCRIPT_FILE"

      # evaluation tuple for SFT (I path only)
      if ! is_lora && [[ $TAG == I ]]; then
        add_eval "${MODEL_BASE}_I_${K}_sft_${LR_TAG}_${suffix_name}" "$OUTDIR"
      fi
    done
  }

  for LR in "${learning_rates[@]}"; do
    generate_train "$B_MODEL" B "$LR"
    generate_train "$I_MODEL" I "$LR"
  done

  # ================= delta-merge =================
  if is_lora; then
    echo "##### LoRA delta-merge #####"                  >> "$SCRIPT_FILE"
    merge_lora() {
      local SRC=$1 SRC_TAG=$2 TGT=$3 TGT_TAG=$4 LR=$5
      local DEC=$(sci2dec "$LR") LR_TAG="lr${DEC}-${suffix_name}"
      local SRC_ROOT="$RESULTS_DIR/${MONTHDAY}/result-${MODEL_BASE}-${MONTHDAY}"

      for RANK in "${lora_ranks[@]}"; do
        for MAX in "${samples[@]}"; do
          local K="$(awk "BEGIN{printf \"%.1f\", $MAX/1000}")k"
          local ADAP="$SRC_ROOT/${SRC_TAG}-${K}-lora-rank${RANK}-ratio${ratios[0]}-${LR_TAG}"
          local TAG="${SRC_TAG}2${TGT_TAG}"
          local MERGED="$SRC_ROOT/merged-${TAG}"

          echo "python3 $WORKSPACE_DIR/src/shadow/merge_lora.py \\" >> "$SCRIPT_FILE"
          echo "  --adapter_path \"$ADAP\" \\"                              >> "$SCRIPT_FILE"
          echo "  --target_base \"$TGT\" \\"                                >> "$SCRIPT_FILE"
          echo "  --merge_tag \"$TAG\" \\"                                  >> "$SCRIPT_FILE"
          echo "  --template \"$template\""                                 >> "$SCRIPT_FILE"
          echo ""                                                           >> "$SCRIPT_FILE"

          add_eval "${MODEL_BASE}_merged_${TAG}_lora${RANK}_lr${DEC}_${suffix_name}" "$MERGED"
        done
      done
    }

    for LR in "${learning_rates[@]}"; do
      merge_lora "$B_MODEL" B "$I_MODEL" I "$LR"
      merge_lora "$I_MODEL" I "$I_MODEL" I "$LR"
    done
  else
    echo "##### SFT delta-merge #####"                  >> "$SCRIPT_FILE"
    DEC=$(sci2dec "${learning_rates[0]}")
    LR_TAG="lr${DEC}-${suffix_name}"
    K="$(awk "BEGIN{printf \"%.1f\", ${samples[0]}/1000}")k"
    B_DIR="$RESULTS_DIR/${MONTHDAY}/result-${MODEL_BASE}-${MONTHDAY}/B-${K}-sft-${LR_TAG}"
    MERGED="$RESULTS_DIR/${MONTHDAY}/result-${MODEL_BASE}-${MONTHDAY}/merged-B2I"
    echo "python3 $WORKSPACE_DIR/src/shadow/apply_diff.py \\" >> "$SCRIPT_FILE"
    echo "  --tuned_model \"$B_DIR\" \\"                               >> "$SCRIPT_FILE"
    echo "  --target_model \"$I_MODEL\" \\"                            >> "$SCRIPT_FILE"
    echo "  --base_model \"$B_MODEL\""                                 >> "$SCRIPT_FILE"
    add_eval "${MODEL_BASE}_merged_B2I_lr${DEC}_${suffix_name}" "$MERGED"
  fi

  # ================= evaluation list =================
  echo "##### Evaluation list #####"                    >> "$SCRIPT_FILE"
  for line in "${EVAL_LINES[@]}"; do
    echo "# $line"                                        >> "$SCRIPT_FILE"
  done
  echo ""                                               >> "$SCRIPT_FILE"
done




echo "Generation finished. Check $SCRIPT_FILE"
