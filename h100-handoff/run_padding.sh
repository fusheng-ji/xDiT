#!/bin/bash
# #900 (Ideogram 4) and #901 (HunyuanVideo) on H100: sequence-parallel padding under the
# quantizing attention backends that B200 cannot run (Sage, FA3 FP8), plus FA3.
# usage: run_padding.sh <900|901>. Needs 2 GPUs. Results in $OUT/<pr>.
source "$(dirname "$0")/common.sh"
pr=${1:?usage: run_padding.sh 900|901}
O=$OUT/$pr; mkdir -p "$O"
env_report > "$O/env.txt" 2>&1; cat "$O/env.txt"
{ tree "pr$pr" "$FORK" "h100-handoff-$pr"; tree main "$UPSTREAM" main; } | tee "$O/heads.txt"
export OMP_NUM_THREADS=16  # torchrun's default of 1 makes the CPU-side FP8 weight conversion crawl

if [ "$pr" = 900 ]; then
  M=xfuser.model_executor.models.transformers.transformer_ideogram4
  # 1024x1024: 4287 text + image tokens, so two ranks pad one.
  A=(--model ideogram-ai/ideogram-4-fp8 --prompt "A minimalist poster for a summer jazz festival, bold typography that reads 'JAZZ NIGHTS', warm sunset colors" --height 1024 --width 1024 --num_inference_steps 20 --seed 42)
else
  M=xfuser.model_executor.models.transformers.transformer_hunyuan_video
  # 848x464, 9 frames: 4611 video tokens, so two ranks pad one.
  A=(--model HunyuanVideo --prompt "A cat walks on the grass, realistic style." --height 464 --width 848 --num_frames 9 --num_inference_steps 4 --seed 42 --enable_tiling)
fi

# run <name> <tree> <nproc> <extra env> runner-args...
port=29700
run() {
  local name=$1 t=$2 n=$3 extra=$4; shift 4
  local d=$O/$name; rm -rf "$d"; mkdir -p "$d"; port=$((port + 1))
  ( cd "$TREES/$t" && echo "head $(git rev-parse HEAD)" > "$d/log.txt"
    env $extra PYTHONPATH="$TREES/$t" PAD_LOG_MODULE=$M timeout 5400 "$PY" -m torch.distributed.run \
      --nproc_per_node="$n" --master_port=$port "$HANDOFF/drive.py" --output_directory "$d" "$@" >> "$d/log.txt" 2>&1
    echo "exit $?" >> "$d/log.txt" )
  echo "$name: $(tail -n 1 "$d/log.txt")"
}
for be in SAGE FLASH_3_FP8 FLASH_3; do
  B=(--attention_backend "$be")
  run "$be-1gpu" "pr$pr" 1 "" "${A[@]}" "${B[@]}"
  run "$be-pr-u2" "pr$pr" 2 "" "${A[@]}" "${B[@]}" --ulysses_degree 2
  run "$be-nozero-u2" "pr$pr" 2 "NO_PAD_ZERO=1" "${A[@]}" "${B[@]}" --ulysses_degree 2
  run "$be-main-u2" main 2 "" "${A[@]}" "${B[@]}" --ulysses_degree 2
  run "$be-pr-r2" "pr$pr" 2 "" "${A[@]}" "${B[@]}" --ring_degree 2
  run "$be-main-r2" main 2 "" "${A[@]}" "${B[@]}" --ring_degree 2
done | tee "$O/exits.txt"

# PSNR of every two-GPU run against the one-GPU run with the same backend (nothing is padded there).
for be in SAGE FLASH_3_FP8 FLASH_3; do
  for c in pr-u2 nozero-u2 main-u2 pr-r2 main-r2; do
    "$PY" "$HANDOFF/cmp.py" "$O/$be-$c" "$O/$be-1gpu" 2>&1 | tail -n 1
  done
  "$PY" "$HANDOFF/cmp.py" "$O/$be-pr-u2" "$O/$be-nozero-u2" 2>&1 | tail -n 1
done | tee "$O/summary.txt"
echo DONE
