#!/bin/bash
# #856 on H100: FlashAttention 3 serving batched prompts of different lengths, against main.
# Needs 2 GPUs and flash_attn_interface (FA3). Results in $OUT/856.
source "$(dirname "$0")/common.sh"
O=$OUT/856; mkdir -p "$O"
env_report > "$O/env.txt" 2>&1; cat "$O/env.txt"
{ tree pr856 "$FORK" h100-handoff; tree main "$UPSTREAM" main; } | tee "$O/heads.txt"
# The parity script is new in #856; main gets a copy so the same script drives both.
cp "$TREES/pr856/tests/e2e/batched_prompt_masks.py" "$TREES/main/tests/e2e/batched_prompt_masks.py"

# e2e <tree> <model> <ulysses> <backend>: the wrapper on <ulysses> GPUs against stock diffusers on one.
e2e() {
  local t=$1 m=$2 u=$3 be=$4 log=$O/e2e-$1-$2-u$3-$4.log
  ( cd "$TREES/$t" && PYTHONPATH="$TREES/$t" timeout 3600 "$PY" tests/e2e/batched_prompt_masks.py \
      --model "$m" --ulysses-degree "$u" --attention-backend "$be" --dtype bfloat16 --tolerance 100 ) > "$log" 2>&1
  echo "$t $m u$u $be exit=$?"
}
for be in FLASH_3 FLASH; do
  for m in z-image qwen-image hunyuan-video-1.5; do
    for u in 1 2; do
      e2e pr856 "$m" "$u" "$be"
      e2e main "$m" "$u" "$be"
    done
  done
done | tee "$O/e2e-exits.txt"

# Accelerator tests on the PR head: one device, then the multi-GPU padding parity file.
( cd "$TREES/pr856" && export PYTHONPATH="$TREES/pr856"
  CUDA_VISIBLE_DEVICES=0 timeout 3600 "$PY" -m pytest -q -rfE -p no:cacheprovider \
    -m "accelerator and not rocm and not multi_gpu" > "$O/pytest-1gpu.txt" 2>&1
  timeout 5400 "$PY" -m pytest -v -rfE -p no:cacheprovider -m "accelerator and multi_gpu and not rocm" \
    tests/integration/accelerator/model_executor/models/test_sequence_parallel_padding.py \
    tests/integration/accelerator/model_executor/test_qwen_batched_prompts_ulysses.py > "$O/pytest-multi.txt" 2>&1 )
tail -n 1 "$O/pytest-1gpu.txt" "$O/pytest-multi.txt"
"$PY" "$HANDOFF/summarize_856.py" "$O" | tee "$O/summary.md"
echo DONE
