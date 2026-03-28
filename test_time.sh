#!/usr/bin/env bash
set -euo pipefail

############################
# User config
############################

OUTDIR="benchmark_compare"
mkdir -p "${OUTDIR}"

PYTHON_BIN=python

MODEL_DIR="./models/stable-diffusion-3.5-medium"
SPEC_PATH="/data2/toby/OSCAR/scripts/temporary.json"
OPENAI_CLIP_JIT_PATH="${HOME}/.cache/clip/ViT-B-32.pt"

PROMPT="a photo of a potted plant"
NEGATIVE_PROMPT="low quality, blurry, repetitive"

G=32
STEPS=30
GUIDANCES=(3.0 5.0 7.5)

# 如果你只想测一个 guidance，就改成：
# GUIDANCES=(3.0)

############################
# Helper functions
############################

monitor_gpus() {
  local gpu_list="$1"
  local outfile="$2"

  # gpu_list 例如: "2,3" 或 "0"
  while true; do
    nvidia-smi \
      --query-gpu=index,timestamp,memory.used,utilization.gpu,utilization.memory \
      --format=csv,noheader,nounits \
      -i "${gpu_list}" >> "${outfile}"
    sleep 0.2
  done
}

summarize_gpu_log() {
  local logfile="$1"

  if [[ ! -s "${logfile}" ]]; then
    echo "0,0,0"
    return
  fi

  # 输出:
  # total_peak_mem_mib, avg_gpu_util_pct, avg_mem_util_pct
  awk -F',' '
    BEGIN {
      sum_gpu=0; sum_memutil=0; n=0;
    }
    {
      # 1=index, 2=timestamp, 3=memory.used, 4=utilization.gpu, 5=utilization.memory
      gsub(/^ +| +$/, "", $1)
      gsub(/^ +| +$/, "", $3)
      gsub(/^ +| +$/, "", $4)
      gsub(/^ +| +$/, "", $5)

      idx=$1
      mem=$3+0
      gpu=$4+0
      memutil=$5+0

      if (mem > peak_mem[idx]) peak_mem[idx]=mem

      sum_gpu += gpu
      sum_memutil += memutil
      n += 1
    }
    END {
      total_peak=0
      for (k in peak_mem) total_peak += peak_mem[k]

      if (n == 0) {
        printf "0,0,0\n"
      } else {
        printf "%.2f,%.2f,%.2f\n", total_peak, sum_gpu/n, sum_memutil/n
      }
    }
  ' "${logfile}"
}

run_one() {
  local method="$1"
  local guidance="$2"
  local visible_gpus="$3"
  shift 3
  local -a cmd=( "$@" )

  local safe_guidance="${guidance//./p}"
  local run_dir="${OUTDIR}/${method}_g${safe_guidance}"
  mkdir -p "${run_dir}"

  local gpu_log="${run_dir}/gpu_log.csv"
  local time_log="${run_dir}/time_log.txt"
  local stdout_log="${run_dir}/stdout.log"
  local stderr_log="${run_dir}/stderr.log"

  rm -f "${gpu_log}" "${time_log}" "${stdout_log}" "${stderr_log}"

  echo "========================================"
  echo "Method   : ${method}"
  echo "Guidance : ${guidance}"
  echo "GPUs     : ${visible_gpus}"
  echo "Command  : ${cmd[*]}"
  echo "========================================"

  monitor_gpus "${visible_gpus}" "${gpu_log}" &
  local monitor_pid=$!

  set +e
  CUDA_VISIBLE_DEVICES="${visible_gpus}" \
  /usr/bin/time -f "REAL=%e\nUSER=%U\nSYS=%S\nMAXRSS=%M" -o "${time_log}" \
    "${cmd[@]}" > "${stdout_log}" 2> "${stderr_log}"
  local exit_code=$?
  set -e

  kill "${monitor_pid}" >/dev/null 2>&1 || true
  wait "${monitor_pid}" 2>/dev/null || true

  local real_time user_time sys_time maxrss
  real_time=$(grep '^REAL=' "${time_log}" | cut -d= -f2)
  user_time=$(grep '^USER=' "${time_log}" | cut -d= -f2)
  sys_time=$(grep '^SYS=' "${time_log}" | cut -d= -f2)
  maxrss=$(grep '^MAXRSS=' "${time_log}" | cut -d= -f2)

  local gpu_stats total_peak_mem avg_gpu_util avg_mem_util
  gpu_stats=$(summarize_gpu_log "${gpu_log}")
  total_peak_mem=$(echo "${gpu_stats}" | cut -d, -f1)
  avg_gpu_util=$(echo "${gpu_stats}" | cut -d, -f2)
  avg_mem_util=$(echo "${gpu_stats}" | cut -d, -f3)

  echo "${method},${guidance},${visible_gpus},${real_time},${user_time},${sys_time},${maxrss},${total_peak_mem},${avg_gpu_util},${avg_mem_util},${exit_code}" >> "${SUMMARY_CSV}"

  echo "[DONE] ${method} @ guidance=${guidance} | time=${real_time}s | total_peak_gpu_mem=${total_peak_mem} MiB | avg_gpu_util=${avg_gpu_util}%"
}

############################
# Summary CSV
############################

SUMMARY_CSV="${OUTDIR}/summary.csv"
echo "method,guidance,visible_gpus,wall_time_sec,user_time_sec,sys_time_sec,max_cpu_mem_kb,total_peak_gpu_mem_mib,avg_gpu_util_pct,avg_gpu_memutil_pct,exit_code" > "${SUMMARY_CSV}"

############################
# Benchmark loop
############################

for guidance in "${GUIDANCES[@]}"; do
  # -------------------------
  # CADS
  # -------------------------
  run_one "cads" "${guidance}" "2,3" \
    "${PYTHON_BIN}" -u scripts/baseline_cads_json.py \
      --model "${MODEL_DIR}" \
      --spec "${SPEC_PATH}" \
      --G "${G}" \
      --steps "${STEPS}" \
      --guidances "${guidance}" \
      --device-transformer cuda:0 \
      --device-vae cuda:1 \
      --device-clip cuda:0 \
      --method cads

  # -------------------------
  # PG
  # -------------------------
  run_one "pg" "${guidance}" "2" \
    "${PYTHON_BIN}" scripts/baseline_pg_json.py \
      --spec "${SPEC_PATH}" \
      --G "${G}" \
      --steps "${STEPS}" \
      --guidances "${guidance}" \
      --fp16 \
      --device cuda:0 \
      --model "${MODEL_DIR}" \
      --method pg

  # -------------------------
  # DPP
  # -------------------------
  run_one "dpp" "${guidance}" "2,3" \
    "${PYTHON_BIN}" scripts/baseline_dpp.py \
      --prompt "${PROMPT}" \
      --G "${G}" \
      --steps "${STEPS}" \
      --guidance "${guidance}" \
      --fp16 \
      --sd_device cuda:0 \
      --vision_device cuda:0 \
      --model-dir "${MODEL_DIR}" \
      --vision_backend openai_clip_jit \
      --openai_clip_jit_path "${OPENAI_CLIP_JIT_PATH}"

  # -------------------------
  # Our method
  # -------------------------
  # 这里我按 CADS 的双卡风格先写，
  # 你需要把下面这几个参数替换成你代码真实支持的参数名。
  run_one "ourmethod" "${guidance}" "2,3" \
    "${PYTHON_BIN}" -u scripts/ourmethod_json.py \
      --prompt "${PROMPT}" \
      --negative "${NEGATIVE_PROMPT}" \
      --G "${G}" \
      --steps "${STEPS}" \
      --guidance "${guidance}" \
      --model "${MODEL_DIR}" \
      --device-transformer cuda:0 \
      --device-vae cuda:1 \
      --device-clip cuda:0
done

echo
echo "Benchmark finished."
echo "Summary saved to: ${SUMMARY_CSV}"