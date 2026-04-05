#!/bin/bash
set -e

# --- 配置 ---
SPEC="specs/showcase_complex.json"
G_VAL=32
CFG_VAL=5.0
STEPS=30
SEED=1111
MODEL_DIR="./models/stable-diffusion-3.5-medium"
CLIP_PATH="~/.cache/clip/ViT-B-32.pt"
GPUS="0,1"

# --- 1. Base (vanilla SD3.5, no diversity) ---
echo "=== [1/7] Baseline FM-SD3.5 ==="
CUDA_VISIBLE_DEVICES=$GPUS python scripts/baseline_base_t2i.py \
  --spec $SPEC \
  --method base_showcase \
  --model-dir "$MODEL_DIR" \
  --G $G_VAL \
  --guidance $CFG_VAL \
  --steps $STEPS \
  --seed $SEED

# --- 2. OSCAR ---
echo "=== [2/7] OSCAR ==="
CUDA_VISIBLE_DEVICES=$GPUS python scripts/ourmethod_t2i.py \
  --spec $SPEC \
  --method ourmethod_showcase \
  --G $G_VAL \
  --guidance $CFG_VAL \
  --steps $STEPS \
  --seed $SEED \
  --gamma0 0.10 \
  --gamma-max-ratio 0.15

# --- 3. CADS ---
echo "=== [3/7] CADS ==="
CUDA_VISIBLE_DEVICES=$GPUS python scripts/baseline_cads_t2i.py \
  --spec $SPEC \
  --method cads_showcase \
  --G $G_VAL \
  --guidance $CFG_VAL \
  --steps $STEPS \
  --category showcase

# --- 4. DPP (DiverseFlow) ---
echo "=== [4/7] DPP (DiverseFlow) ==="
CUDA_VISIBLE_DEVICES=$GPUS python scripts/baseline_dpp_t2i.py \
  --spec $SPEC \
  --method dpp_showcase \
  --G $G_VAL \
  --guidance $CFG_VAL \
  --steps $STEPS \
  --openai_clip_jit_path "$CLIP_PATH"

# --- 5. PG (Particle Guidance) ---
echo "=== [5/7] PG ==="
CUDA_VISIBLE_DEVICES=$GPUS python scripts/baseline_pg_t2i.py \
  --spec $SPEC \
  --method pg_showcase \
  --model "$MODEL_DIR" \
  --G $G_VAL \
  --cfg $CFG_VAL \
  --steps $STEPS \
  --seed $SEED

# --- 6. APG ---
echo "=== [6/7] APG ==="
CUDA_VISIBLE_DEVICES=$GPUS python scripts/baseline_apg_t2i.py \
  --spec $SPEC \
  --method apg_showcase \
  --model-dir "$MODEL_DIR" \
  --G $G_VAL \
  --guidance $CFG_VAL \
  --steps $STEPS \
  --seed $SEED

# --- 7. Mix ODE/SDE ---
echo "=== [7/7] Mix ODE/SDE ==="
CUDA_VISIBLE_DEVICES=$GPUS python scripts/baseline_mix_flow_t2i.py \
  --spec $SPEC \
  --method mix_showcase \
  --model-dir "$MODEL_DIR" \
  --G $G_VAL \
  --guidance $CFG_VAL \
  --steps $STEPS \
  --seed $SEED

echo "=== All 7 showcase experiments done ==="
echo "Results in: outputs/{base,ourmethod,cads,dpp,pg,apg,mix}_showcase/"
