#!/bin/bash
# GPU 0+1: ourmethod (2-GPU) → base (GPU 0) + mix (GPU 1) in parallel
set -e
source /data2/toby/qdhf/bin/activate

G=64
GUIDANCE=5.0
STEPS=30
MODEL="./models/stable-diffusion-3.5-medium"

cd /data2/toby/OSCAR

echo "=== [GPU 2] pg complex  &  [GPU 3] apg complex  (parallel) ==="
CUDA_VISIBLE_DEVICES=0 python scripts/baseline_pg_t2i.py \
  --spec specs/t2i_complex_mini.json --method pg \
  --model $MODEL --G $G --cfg $GUIDANCE --steps $STEPS --fp16 \
  --device cuda:0 &
PID3=$!

CUDA_VISIBLE_DEVICES=1 python scripts/baseline_apg_t2i.py \
  --spec specs/t2i_complex_mini.json --method apg \
  --model-dir $MODEL --G $G --guidance $GUIDANCE --steps $STEPS --fp16 \
  --device_transformer cuda:0 --device_vae cuda:0 \
  --device_text1 cuda:0 --device_text2 cuda:0 --device_text3 cuda:0 &
PID4=$!

wait $PID3 $PID4

# # ── Phase 1: ourmethod (needs 2 GPUs) ──────────────────────────────────────
# echo "=== [GPU 0+1] ourmethod  color ==="
# CUDA_VISIBLE_DEVICES=0,1 python scripts/ourmethod_t2i.py \
#   --spec specs/t2i_color_mini.json --method ourmethod \
#   --G $G --guidance $GUIDANCE --steps $STEPS

# echo "=== [GPU 0+1] ourmethod  complex ==="
# CUDA_VISIBLE_DEVICES=0,1 python scripts/ourmethod_t2i.py \
#   --spec specs/t2i_complex_mini.json --method ourmethod \
#   --G $G --guidance $GUIDANCE --steps $STEPS

# # ── Phase 2: base on GPU 0  &  mix on GPU 1  (parallel, single-GPU each) ───
# echo "=== [GPU 0] base color  &  [GPU 1] mix color  (parallel) ==="
# CUDA_VISIBLE_DEVICES=0 python scripts/baseline_base_t2i.py \
#   --spec specs/t2i_color_mini.json --method base \
#   --model-dir $MODEL --G $G --guidance $GUIDANCE --steps $STEPS --fp16 \
#   --device_transformer cuda:0 --device_vae cuda:0 \
#   --device_text1 cuda:0 --device_text2 cuda:0 --device_text3 cuda:0 &
# PID1=$!

# CUDA_VISIBLE_DEVICES=1 python scripts/baseline_mix_flow_t2i.py \
#   --spec specs/t2i_color_mini.json --method mix \
#   --model-dir $MODEL --G $G --guidance $GUIDANCE --steps $STEPS --fp16 \
#   --device_transformer cuda:0 --device_vae cuda:0 \
#   --device_text1 cuda:0 --device_text2 cuda:0 --device_text3 cuda:0 &
# PID2=$!

# wait $PID1 $PID2

# echo "=== [GPU 0] base complex  &  [GPU 1] mix complex  (parallel) ==="
# CUDA_VISIBLE_DEVICES=0 python scripts/baseline_base_t2i.py \
#   --spec specs/t2i_complex_mini.json --method base \
#   --model-dir $MODEL --G $G --guidance $GUIDANCE --steps $STEPS --fp16 \
#   --device_transformer cuda:0 --device_vae cuda:0 \
#   --device_text1 cuda:0 --device_text2 cuda:0 --device_text3 cuda:0 &
# PID3=$!

# CUDA_VISIBLE_DEVICES=1 python scripts/baseline_mix_flow_t2i.py \
#   --spec specs/t2i_complex_mini.json --method mix \
#   --model-dir $MODEL --G $G --guidance $GUIDANCE --steps $STEPS --fp16 \
#   --device_transformer cuda:0 --device_vae cuda:0 \
#   --device_text1 cuda:0 --device_text2 cuda:0 --device_text3 cuda:0 &
# PID4=$!

# wait $PID3 $PID4

echo "=== Pair 01 ALL DONE ==="
