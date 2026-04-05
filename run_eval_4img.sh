#!/bin/bash
# Run 4-image evaluation in parallel across 4 GPUs
# Each GPU handles ~2 methods; all 3 concepts per method

cd /data2/toby/OSCAR
source /data2/toby/qdhf/bin/activate

CONCEPTS="t2i_color t2i_complex t2i_spatial"
N=4
ROOT=./outputs

mkdir -p logs

echo "=== Starting 4-image eval (N=$N) on 4 GPUs ==="

# GPU 0: ourmethod
CUDA_VISIBLE_DEVICES=0 python eval_4img.py \
    --outputs-root $ROOT \
    --methods ourmethod \
    --concepts $CONCEPTS \
    --n-imgs $N --seed 42 --device cuda:0 \
    2>&1 | tee logs/eval4img_gpu0.log &
PID0=$!

# GPU 1: base + mix
CUDA_VISIBLE_DEVICES=1 python eval_4img.py \
    --outputs-root $ROOT \
    --methods base mix \
    --concepts $CONCEPTS \
    --n-imgs $N --seed 42 --device cuda:0 \
    2>&1 | tee logs/eval4img_gpu1.log &
PID1=$!

# GPU 2: cads + dpp
CUDA_VISIBLE_DEVICES=2 python eval_4img.py \
    --outputs-root $ROOT \
    --methods cads dpp \
    --concepts $CONCEPTS \
    --n-imgs $N --seed 42 --device cuda:0 \
    2>&1 | tee logs/eval4img_gpu2.log &
PID2=$!

# GPU 3: pg + apg
CUDA_VISIBLE_DEVICES=3 python eval_4img.py \
    --outputs-root $ROOT \
    --methods pg apg \
    --concepts $CONCEPTS \
    --n-imgs $N --seed 42 --device cuda:0 \
    2>&1 | tee logs/eval4img_gpu3.log &
PID3=$!

echo "PIDs: $PID0 $PID1 $PID2 $PID3"
echo "Waiting for all GPUs to finish..."
wait $PID0 $PID1 $PID2 $PID3

echo "=== All eval done. Running analysis... ==="
python eval_4img.py \
    --outputs-root $ROOT \
    --methods ourmethod base cads dpp pg apg mix \
    --concepts $CONCEPTS \
    --n-imgs $N --seed 42 --device cuda:0 \
    --top-k 30 \
    --skip-eval \
    2>&1 | tee logs/eval4img_analysis.log

echo "=== Done! Check outputs/best_prompts_4img_top30.csv ==="
