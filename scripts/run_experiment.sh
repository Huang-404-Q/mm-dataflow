#!/usr/bin/env bash
# Run the A/B/C fine-tuning experiment end to end.
#
#   bash scripts/run_experiment.sh                 # all three groups
#   bash scripts/run_experiment.sh b_cleaned       # just one
#
# Written to survive a rented GPU disappearing mid-experiment: each group's
# completion is marked by its adapter file, and a re-run skips whatever already
# finished. Interrupting after group A and resuming an hour later costs nothing.
#
# Every group is launched from the SAME config (train/qwen2_5vl_lora_sft.yaml)
# with only `dataset` and `output_dir` overridden -- see the header of that file
# for why there is no per-group YAML.
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG=train/qwen2_5vl_lora_sft.yaml
SFT_DIR=${SFT_DIR:-data/sft}
OUT_ROOT=${OUT_ROOT:-outputs/sft}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}
if [[ $# -gt 0 ]]; then GROUPS=("$@"); else GROUPS=(a_dirty b_cleaned c_random); fi

mkdir -p "$OUT_ROOT" logs

if [[ ! -f "$SFT_DIR/dataset_info.json" ]]; then
  echo "!! $SFT_DIR/dataset_info.json missing -- run scripts/make_sft_sets.py first" >&2
  exit 1
fi

echo "== A/B/C experiment"
echo "   config : $CONFIG"
echo "   groups : ${GROUPS[*]}"
echo "   model  : $BASE_MODEL"
echo

for g in "${GROUPS[@]}"; do
  out="$OUT_ROOT/$g"
  # adapter_model.safetensors is written only on a clean finish, so its presence
  # is a real completion marker; a directory left by a crashed run is not.
  if [[ -f "$out/adapter_model.safetensors" ]]; then
    echo "-- $g: already trained, skipping"
    continue
  fi

  echo "-- $g: training -> $out"
  start=$(date +%s)
  llamafactory-cli train "$CONFIG" \
      dataset="mmdf_$g" \
      dataset_dir="$SFT_DIR" \
      output_dir="$out" \
    2>&1 | tee "logs/train_$g.log"
  echo "   done in $(( ($(date +%s) - start) / 60 )) min"
done

# Merge for serving. vLLM can load adapters at runtime, but the throughput
# benchmark should measure the model that would actually be deployed, and a
# merged checkpoint removes the adapter-dispatch overhead from the numbers.
for g in "${GROUPS[@]}"; do
  out="$OUT_ROOT/$g"
  merged="$OUT_ROOT/${g}_merged"
  if [[ ! -f "$out/adapter_model.safetensors" ]]; then continue; fi
  if [[ -f "$merged/config.json" ]]; then echo "-- $g: already merged"; continue; fi

  echo "-- $g: merging LoRA -> $merged"
  llamafactory-cli export \
      --model_name_or_path "$BASE_MODEL" \
      --adapter_name_or_path "$out" \
      --template qwen2_vl \
      --finetuning_type lora \
      --export_dir "$merged" \
      --export_size 5 \
      --trust_remote_code true \
    2>&1 | tee "logs/merge_$g.log"
done

echo
echo "== done. next:"
echo "   python scripts/run_eval.py --models base $OUT_ROOT/*_merged --out docs/results.md"
echo "   python scripts/bench_serving.py --model $OUT_ROOT/b_cleaned_merged"
