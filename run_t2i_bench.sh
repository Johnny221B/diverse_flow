#!/bin/bash

# 确保遇到错误就停止
set -e

# --- 统一参数配置 ---
G_VAL=32
CFG_VAL=5.0
STEPS=50
MODEL_DIR="./models/stable-diffusion-3.5-medium"
CLIP_PATH="~/.cache/clip/ViT-B-32.pt"
GPUS="0,1"

# 定义运行函数，减少冗余
run_experiment() {
    local script=$1
    local spec=$2
    local method=$3
    echo ">>> Running $method on $spec <<<"
    CUDA_VISIBLE_DEVICES=$GPUS python $script \
      --spec $spec \
      --method $method \
      --model-dir "$MODEL_DIR" \
      --G $G_VAL \
      --guidance $CFG_VAL \
      --steps $STEPS
}

# --- 1. Our Method (OSCAR) ---
# echo "=== 运行 Our Method 测试 ==="
# CUDA_VISIBLE_DEVICES=$GPUS python scripts/ourmethod_t2i.py --spec specs/t2i_human_mini.json --method ourmethod --G $G_VAL --guidance $CFG_VAL  --steps $STEPS
# CUDA_VISIBLE_DEVICES=$GPUS python scripts/ourmethod_t2i.py --spec specs/t2i_complex_mini.json --method ourmethod --G $G_VAL --guidance $CFG_VAL
# CUDA_VISIBLE_DEVICES=$GPUS python scripts/ourmethod_t2i.py --spec specs/t2i_spatial_mini.json --method ourmethod --G $G_VAL --guidance $CFG_VAL

# --- 2. DPP Baseline ---
# echo "=== 运行 DPP Baseline 测试 ==="
# 注意：确保你的脚本里参数名是 --guidance 而不是 --guidances
# CUDA_VISIBLE_DEVICES=$GPUS python scripts/baseline_dpp_t2i.py --spec specs/t2i_human_mini.json --method dpp --G $G_VAL --guidance $CFG_VAL --openai_clip_jit_path "$CLIP_PATH" --steps $STEPS
# CUDA_VISIBLE_DEVICES=$GPUS python scripts/baseline_dpp_t2i.py --spec specs/t2i_complex_mini.json --method dpp_t2i --G $G_VAL --guidance $CFG_VAL --openai_clip_jit_path "$CLIP_PATH"
# CUDA_VISIBLE_DEVICES=$GPUS python scripts/baseline_dpp_t2i.py --spec specs/t2i_spatial_mini.json --method dpp_t2i --G $G_VAL --guidance $CFG_VAL --openai_clip_jit_path "$CLIP_PATH"

echo "=== 运行 CADS Baseline 测试 ==="
CUDA_VISIBLE_DEVICES=$GPUS python scripts/baseline_cads_t2i.py --spec specs/t2i_human_mini.json --method cads --G $G_VAL --guidance $CFG_VAL  --steps $STEPS --category human
# CUDA_VISIBLE_DEVICES=$GPUS python scripts/baseline_cads_t2i.py --spec specs/t2i_color_mini.json --method cads --G $G_VAL --guidance $CFG_VAL  --steps $STEPS --category color
# CUDA_VISIBLE_DEVICES=$GPUS python scripts/baseline_cads_t2i.py --spec specs/t2i_complex_mini.json --method cads --G $G_VAL --guidance $CFG_VAL  --steps $STEPS --category complex
# CUDA_VISIBLE_DEVICES=$GPUS python scripts/baseline_cads_t2i.py --spec specs/t2i_spatial_mini.json --method cads --G $G_VAL --guidance $CFG_VAL  --steps $STEPS --category spatial
# --- 3. PG Baseline ---
# echo "=== 运行 PG Baseline 测试 ==="
# CUDA_VISIBLE_DEVICES=$GPUS python scripts/baseline_pg_t2i.py --spec specs/t2i_color_mini.json --method pg_t2i --model "$MODEL_DIR" --G $G_VAL --cfg $CFG_VAL --steps $STEPS
# CUDA_VISIBLE_DEVICES=$GPUS python scripts/baseline_pg_t2i.py --spec specs/t2i_complex_mini.json --method pg_t2i --model "$MODEL_DIR" --G $G_VAL --cfg $CFG_VAL --steps $STEPS
# CUDA_VISIBLE_DEVICES=$GPUS python scripts/baseline_pg_t2i.py --spec specs/t2i_spatial_mini.json --method pg_t2i --model "$MODEL_DIR" --G $G_VAL --cfg $CFG_VAL --steps $STEPS

echo "=== 所有 T2I-CompBench 实验指令已提交完成 ==="