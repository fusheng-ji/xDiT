#!/bin/bash
# #856 on H100: latency and peak memory of padded prompt batches, main against the PR.
# Each config runs main, PR, main, so the spread between the two main runs bounds the noise.
# Needs 2 GPUs and flash_attn_interface (FA3). Results in $OUT/856-timing.
source "$(dirname "$0")/common.sh"
O=$OUT/856-timing; mkdir -p "$O"
env_report > "$O/env.txt" 2>&1; cat "$O/env.txt"
{ tree pr856 "$FORK" h100-handoff; tree main "$UPSTREAM" main; } | tee "$O/heads.txt"
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv >> "$O/env.txt" 2>&1

SHORT="A red fox in the snow."
LONG="A weathered fisherman mends a bright orange net on the wooden pier of a small harbor at sunrise, gulls circling above the moored boats, soft golden light, misty water, detailed photograph, 35mm film."
LONGER="$LONG A small red lighthouse stands on the far headland, wet planks, rope coils and crates."
IMG="--height 1024 --width 1024 --num_inference_steps 20"
GPUS=${CUDA_VISIBLE_DEVICES:-0,1}; FIRST=${GPUS%%,*}
PORT=29700

# run <name> <tree> <gpus> <model> <runner args...>
run() {
  local name=$1 t=$2 n=$3 model=$4; shift 4
  local d=$O/$name; rm -rf "$d"; mkdir -p "$d"
  local dev=$FIRST; [ "$n" = 2 ] && dev=$GPUS
  PORT=$((PORT + 1))
  ( cd "$TREES/$t" && PYTHONPATH="$TREES/$t" CUDA_VISIBLE_DEVICES=$dev timeout 1800 \
      "$PY" -m torch.distributed.run --nproc_per_node="$n" --master_port=$PORT "$HANDOFF/peak.py" \
      --model "$model" --seed 42 --warmup_calls 1 --num_iterations 3 --output_directory "$d" "$@" ) > "$d/log.txt" 2>&1
  echo "$name exit=$?"
}
for be in FLASH_3 FLASH; do
  for u in 1 2; do
    ua=""; [ "$u" = 2 ] && ua="--ulysses_degree 2"
    for pass in a b c; do
      t=main; [ "$pass" = b ] && t=pr856
      run "Z1p_u${u}_${be}_${pass}_$t" "$t" "$u" Z-Image $IMG --prompt "$LONGER" --attention_backend "$be" $ua
      run "Z2p_u${u}_${be}_${pass}_$t" "$t" "$u" Z-Image $IMG --prompt "$SHORT" "$LONG" --attention_backend "$be" $ua
      run "Q2p_u${u}_${be}_${pass}_$t" "$t" "$u" Qwen-Image $IMG --prompt "$SHORT" "$LONG" --attention_backend "$be" $ua
    done
  done
done | tee "$O/exits.txt"
rm -f "$O"/*/*.png
"$PY" "$HANDOFF/summarize_timing.py" "$O" | tee "$O/summary.md"
echo DONE
