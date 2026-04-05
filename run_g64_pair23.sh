#!/bin/bash
# GPU 2+3: dpp (2-GPU) → cads (2-GPU) → pg (GPU 2) + apg (GPU 3) in parallel
set -e
source /data2/toby/qdhf/bin/activate

G=64
GUIDANCE=5.0
STEPS=30
MODEL="./models/stable-diffusion-3.5-medium"
CLIP_PATH="$HOME/.cache/clip/ViT-B-32.pt"

cd /data2/toby/OSCAR

# ── Phase 1: dpp (needs 2 GPUs, CUDA_VISIBLE=2,3 → cuda:0=GPU2, cuda:1=GPU3) ──
echo "=== [GPU 2+3] dpp  color ==="
CUDA_VISIBLE_DEVICES=2,3 python scripts/baseline_dpp_t2i.py \
  --spec specs/t2i_color_mini.json --method dpp \
  --G $G --guidance $GUIDANCE --steps $STEPS \
  --openai_clip_jit_path "$CLIP_PATH"

echo "=== [GPU 2+3] dpp  complex ==="
CUDA_VISIBLE_DEVICES=2,3 python scripts/baseline_dpp_t2i.py \
  --spec specs/t2i_complex_mini.json --method dpp \
  --G $G --guidance $GUIDANCE --steps $STEPS \
  --openai_clip_jit_path "$CLIP_PATH"

# ── Phase 2: cads (needs 2 GPUs) ───────────────────────────────────────────
echo "=== [GPU 2+3] cads  color ==="
CUDA_VISIBLE_DEVICES=2,3 python scripts/baseline_cads_t2i.py \
  --spec specs/t2i_color_mini.json --method cads \
  --model-dir $MODEL --G $G --guidance $GUIDANCE --steps $STEPS \
  --category color

echo "=== [GPU 2+3] cads  complex ==="
CUDA_VISIBLE_DEVICES=2,3 python scripts/baseline_cads_t2i.py \
  --spec specs/t2i_complex_mini.json --method cads \
  --model-dir $MODEL --G $G --guidance $GUIDANCE --steps $STEPS \
  --category complex

# ── Phase 3: pg on GPU 2  &  apg on GPU 3  (parallel, single-GPU each) ────
echo "=== [GPU 2] pg color  &  [GPU 3] apg color  (parallel) ==="
CUDA_VISIBLE_DEVICES=2 python scripts/baseline_pg_t2i.py \
  --spec specs/t2i_color_mini.json --method pg \
  --model $MODEL --G $G --cfg $GUIDANCE --steps $STEPS --fp16 \
  --device cuda:0 &
PID1=$!

CUDA_VISIBLE_DEVICES=3 python scripts/baseline_apg_t2i.py \
  --spec specs/t2i_color_mini.json --method apg \
  --model-dir $MODEL --G $G --guidance $GUIDANCE --steps $STEPS --fp16 \
  --device_transformer cuda:0 --device_vae cuda:0 \
  --device_text1 cuda:0 --device_text2 cuda:0 --device_text3 cuda:0 &
PID2=$!

wait $PID1 $PID2

echo "=== [GPU 2] pg complex  &  [GPU 3] apg complex  (parallel) ==="
CUDA_VISIBLE_DEVICES=2 python scripts/baseline_pg_t2i.py \
  --spec specs/t2i_complex_mini.json --method pg \
  --model $MODEL --G $G --cfg $GUIDANCE --steps $STEPS --fp16 \
  --device cuda:0 &
PID3=$!

CUDA_VISIBLE_DEVICES=3 python scripts/baseline_apg_t2i.py \
  --spec specs/t2i_complex_mini.json --method apg \
  --model-dir $MODEL --G $G --guidance $GUIDANCE --steps $STEPS --fp16 \
  --device_transformer cuda:0 --device_vae cuda:0 \
  --device_text1 cuda:0 --device_text2 cuda:0 --device_text3 cuda:0 &
PID4=$!

wait $PID3 $PID4

echo "=== Pair 23 ALL DONE ==="
