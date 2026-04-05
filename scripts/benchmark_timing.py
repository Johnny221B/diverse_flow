#!/usr/bin/env python3
"""
Timing & peak-VRAM benchmark for all 7 T2I methods.

Fixed variables across all runs:
  prompt  : "a blue bicycle and a red chair"
  steps   : 30
  height  : 512
  width   : 512
  guidance: 5.0
  seed    : 1111

Batch sizes tested: 4 and 32

GPU layout (noted in results):
  1-GPU methods  (cuda:0)        : base, apg, mix, pg
  2-GPU methods  (cuda:0+cuda:1) : cads, dpp, ourmethod

Fairness notes:
  - Use --warmup to run a cheap G=1,steps=5 pass before timing each method.
    This pre-loads model weights into OS page cache so timed runs don't include
    cold-disk I/O, making sequential benchmarks comparable.
  - 1-GPU vs 2-GPU hardware difference is noted per-method; 2-GPU methods may
    be faster because SD3.5 transformer on GPU1 has full bandwidth.

Usage:
  python scripts/benchmark_timing.py                      # all methods, G=4,32
  python scripts/benchmark_timing.py --methods ourmethod base --G 4 32
  python scripts/benchmark_timing.py --warmup             # pre-warm cache first
  python scripts/benchmark_timing.py --dry-run            # print commands only
"""

from __future__ import annotations
import argparse
import subprocess
import sys
import os
import time
import threading
import json
from pathlib import Path

# ─── Repo / model paths ────────────────────────────────────────────────────
REPO   = Path(__file__).resolve().parent.parent
MODEL  = str(REPO / "models" / "stable-diffusion-3.5-medium")
CLIP   = os.path.expanduser("~/.cache/clip/ViT-B-32.pt")
SCRIPT = REPO / "scripts"
OUTDIR = REPO / "outputs" / "_benchmark_timing"
SPEC   = str(REPO / "specs" / "benchmark_single.json")   # {"_bench":["<PROMPT>"]}

# Python interpreter inside the project virtualenv
VENV_PY = str(REPO.parent / "qdhf" / "bin" / "python")

# ─── Shared generation constants ───────────────────────────────────────────
PROMPT   = "a blue bicycle and a red chair"
STEPS    = 30
HEIGHT   = 512
WIDTH    = 512
GUIDANCE = 5.0
SEED     = 1111

# GPU count per method (for reporting; does not change the run)
# All methods use 2 GPUs.  pg runs as two parallel G//2 sub-processes (one per GPU)
# because FlowSampler loads the full model onto a single device.
GPU_COUNT = {
    "base": 2, "apg": 2, "mix": 2, "pg": 2,
    "cads": 2, "dpp": 2, "ourmethod": 2, "ourmethod_fast": 2,
}


def build_cmd(method: str, G: int, gpu_vae: int = 0, gpu_tr: int = 1,
              warmup: bool = False) -> list[str]:
    """Return the subprocess command list for a given method and batch size.

    gpu_vae: GPU index for VAE (and CLIP for dpp/ourmethod)
    gpu_tr:  GPU index for transformer + text encoders
    warmup=True: tiny G=1, steps=5 run to pre-load model weights into OS cache.
    """
    py = VENV_PY if os.path.isfile(VENV_PY) else sys.executable
    exe = [py]
    # Use a _wu suffix for warmup so its outputs go to a separate directory
    # and never trigger the skip-if-complete check on the timed run.
    m_tag = f"_bench_{method}_G{G}_wu" if warmup else f"_bench_{method}_G{G}"

    _G     = G    if warmup else G      # same batch size so CuDNN kernels for that shape get compiled
    _steps = 3    if warmup else STEPS  # only 3 steps to minimise warmup time
    _seed  = 9999 if warmup else SEED

    d_vae = f"cuda:{gpu_vae}"
    d_tr  = f"cuda:{gpu_tr}"

    if method == "base":
        return exe + [
            str(SCRIPT / "baseline_base_t2i.py"),
            "--spec", SPEC, "--method", m_tag, "--model-dir", MODEL,
            "--G", str(_G), "--steps", str(_steps),
            "--guidance", str(GUIDANCE), "--seed", str(_seed),
            "--device_transformer", d_tr, "--device_vae", d_vae,
            "--device_text1", d_tr, "--device_text2", d_tr, "--device_text3", d_tr,
        ]

    if method == "apg":
        return exe + [
            str(SCRIPT / "baseline_apg_t2i.py"),
            "--spec", SPEC, "--method", m_tag, "--model-dir", MODEL,
            "--G", str(_G), "--steps", str(_steps),
            "--guidance", str(GUIDANCE), "--seed", str(_seed),
            "--device_transformer", d_tr, "--device_vae", d_vae,
            "--device_text1", d_tr, "--device_text2", d_tr, "--device_text3", d_tr,
        ]

    if method == "mix":
        return exe + [
            str(SCRIPT / "baseline_mix_flow_t2i.py"),
            "--spec", SPEC, "--method", m_tag, "--model-dir", MODEL,
            "--G", str(_G), "--steps", str(_steps),
            "--guidance", str(GUIDANCE), "--seed", str(_seed),
            "--device_transformer", d_tr, "--device_vae", d_vae,
            "--device_text1", d_tr, "--device_text2", d_tr, "--device_text3", d_tr,
        ]

    if method == "pg":
        # Handled by run_pg_parallel; this path is kept for warmup only.
        # gpu_vae is used as the single device for the warmup pass.
        return exe + [
            str(SCRIPT / "baseline_pg_t2i.py"),
            "--prompt", PROMPT, "--method", m_tag + "_warmup", "--model", MODEL,
            "--G", str(_G), "--steps", str(_steps),
            "--cfg", str(GUIDANCE), "--seed", str(_seed),
            "--device", d_vae,
            "--height", str(HEIGHT), "--width", str(WIDTH),
        ]

    if method == "cads":
        return exe + [
            str(SCRIPT / "baseline_cads_t2i.py"),
            "--spec", SPEC, "--method", m_tag, "--model-dir", MODEL,
            "--G", str(_G), "--steps", str(_steps),
            "--guidance", str(GUIDANCE), "--seed", str(_seed),
            "--device_transformer", d_tr, "--device_vae", d_vae,
            "--device_text1", d_tr, "--device_text2", d_tr, "--device_text3", d_tr,
        ]

    if method == "dpp":
        # dpp does not accept --height/--width; hardcodes 512×512 internally
        return exe + [
            str(SCRIPT / "baseline_dpp_t2i.py"),
            "--prompt", PROMPT, "--method", m_tag, "--model-dir", MODEL,
            "--G", str(_G), "--steps", str(_steps),
            "--guidance", str(GUIDANCE), "--seed", str(_seed),
            "--device_transformer", d_tr, "--device_vae", d_vae, "--device_clip", d_vae,
            "--openai_clip_jit_path", CLIP,
        ]

    if method == "ourmethod":
        return exe + [
            str(SCRIPT / "ourmethod_t2i.py"),
            "--prompt", PROMPT, "--method", m_tag, "--model-dir", MODEL,
            "--G", str(_G), "--steps", str(_steps),
            "--guidance", str(GUIDANCE), "--seed", str(_seed),
            "--clip-jit", CLIP,
            "--device-transformer", d_tr, "--device-vae", d_vae, "--device-clip", d_vae,
            "--height", str(HEIGHT), "--width", str(WIDTH),
        ]

    if method == "ourmethod_fast":
        return exe + [
            str(SCRIPT / "ourmethod_t2i.py"),
            "--prompt", PROMPT, "--method", m_tag, "--model-dir", MODEL,
            "--G", str(_G), "--steps", str(_steps),
            "--guidance", str(GUIDANCE), "--seed", str(_seed),
            "--clip-jit", CLIP,
            "--device-transformer", d_tr, "--device-vae", d_vae, "--device-clip", d_vae,
            "--height", str(HEIGHT), "--width", str(WIDTH),
            "--update-every", "3",
        ]

    raise ValueError(f"Unknown method: {method}")


# ─── VRAM sampling ─────────────────────────────────────────────────────────

class VRAMMonitor:
    """Background thread: polls `nvidia-smi` every 0.5s, records peak per GPU."""

    def __init__(self, gpu_ids: list[int], interval: float = 0.5):
        self.gpu_ids   = gpu_ids
        self.interval  = interval
        self.peak_mib  = {g: 0 for g in gpu_ids}
        self._stop     = threading.Event()
        self._thread   = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self):
        while not self._stop.wait(self.interval):
            try:
                out = subprocess.check_output(
                    ["nvidia-smi",
                     "--query-gpu=index,memory.used",
                     "--format=csv,noheader,nounits"],
                    text=True, timeout=3,
                )
                for line in out.strip().splitlines():
                    parts = line.split(",")
                    if len(parts) == 2:
                        idx  = int(parts[0].strip())
                        used = int(parts[1].strip())
                        if idx in self.peak_mib:
                            self.peak_mib[idx] = max(self.peak_mib[idx], used)
            except Exception:
                pass


def gpu_ids_for(method: str, gpu_vae: int = 0, gpu_tr: int = 1) -> list[int]:
    return sorted(set([gpu_vae, gpu_tr]))  # all methods: 2-GPU split


# ─── Benchmark runner ───────────────────────────────────────────────────────

def _run_subprocess(cmd: list[str], label: str, env: dict) -> tuple[bool, str]:
    """Run cmd, return (ok, stderr_tail)."""
    proc = subprocess.run(
        cmd,
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
    )
    ok = (proc.returncode == 0)
    tail = "\n".join(proc.stderr.strip().splitlines()[-20:])
    return ok, tail


def run_once(method: str, G: int, gpu_vae: int = 0, gpu_tr: int = 1,
             dry_run: bool = False, do_warmup: bool = False) -> dict:
    cmd  = build_cmd(method, G, gpu_vae=gpu_vae, gpu_tr=gpu_tr)
    gpus = gpu_ids_for(method, gpu_vae=gpu_vae, gpu_tr=gpu_tr)

    if dry_run:
        print("  CMD:", " ".join(cmd))
        if do_warmup:
            print("  WARMUP CMD:", " ".join(build_cmd(method, G, gpu_vae=gpu_vae, gpu_tr=gpu_tr, warmup=True)))
        return {}

    env = os.environ.copy()
    env["TRANSFORMERS_VERBOSITY"] = "error"
    env["TOKENIZERS_PARALLELISM"] = "false"

    # ── Optional warmup: pre-loads model weights into OS page cache ──────────
    if do_warmup:
        warmup_cmd = build_cmd(method, G, gpu_vae=gpu_vae, gpu_tr=gpu_tr, warmup=True)
        print(f"  [WARMUP] method={method} (G={G} steps=3) ...", flush=True)
        w_ok, w_err = _run_subprocess(warmup_cmd, "warmup", env)
        if not w_ok:
            print(f"  [WARMUP FAIL] {w_err[-200:]}")
        else:
            print(f"  [WARMUP OK]", flush=True)

    # Reset peak-memory counters
    try:
        import torch
        for g in gpus:
            torch.cuda.reset_peak_memory_stats(g)
    except Exception:
        pass

    monitor = VRAMMonitor(gpus, interval=0.5)

    print(f"  [START] method={method} G={G}  GPUs={GPU_COUNT[method]}", flush=True)
    t0 = time.perf_counter()
    monitor.start()

    proc = subprocess.run(
        cmd,
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
    )

    elapsed = time.perf_counter() - t0
    monitor.stop()

    ok = (proc.returncode == 0)
    if not ok:
        print(f"  [FAIL] returncode={proc.returncode}")
        tail = "\n".join(proc.stderr.strip().splitlines()[-20:])
        print(f"  STDERR (tail):\n{tail}")

    peak = monitor.peak_mib
    print(f"  [DONE] elapsed={elapsed:.1f}s  peak_VRAM={peak}  ok={ok}", flush=True)

    return {
        "method":       method,
        "G":            G,
        "elapsed_s":    round(elapsed, 2),
        "ok":           ok,
        "peak_vram_mib": peak,
        "num_gpus":     GPU_COUNT[method],
        "warmup_done":  do_warmup,
    }


def _build_pg_cmd(G_half: int, save_offset: int,
                  warmup: bool = False, full_G: int = 0) -> list[str]:
    """Build a single-GPU PG command for one half of the parallel split.

    Each sub-process is launched with CUDA_VISIBLE_DEVICES set to exactly one
    physical GPU, so the script always uses --device cuda:0 (the only visible GPU).
    """
    py = VENV_PY if os.path.isfile(VENV_PY) else sys.executable
    _G     = G_half  # same batch size for both warmup and timed run
    _steps = 3       if warmup else STEPS
    _seed  = 9999    if warmup else SEED
    # Use _wu suffix for warmup so its output dir never conflicts with the timed run.
    m_tag = f"_bench_pg_G{full_G}_wu" if warmup else f"_bench_pg_G{full_G}"
    return [py,
        str(SCRIPT / "baseline_pg_t2i.py"),
        "--prompt", PROMPT, "--method", m_tag, "--model", MODEL,
        "--G", str(_G), "--steps", str(_steps),
        "--cfg", str(GUIDANCE), "--seed", str(_seed),
        "--device", "cuda:0",   # always 0; CUDA_VISIBLE_DEVICES selects the physical GPU
        "--height", str(HEIGHT), "--width", str(WIDTH),
        "--save-offset", str(save_offset),
    ]


def run_pg_parallel(G: int, gpu_vae: int = 0, gpu_tr: int = 1,
                    dry_run: bool = False, do_warmup: bool = False) -> dict:
    """Run PG as two G//2 sub-processes in parallel, one per GPU.

    FlowSampler loads the full SD3.5 model onto a single device, so we achieve
    2-GPU fairness by running two independent half-batch processes simultaneously.
    Both write to the same output directory with non-overlapping file indices
    (e.g. 000-001 on GPU0, 002-003 on GPU1 for G=4).
    """
    G_half = max(1, G // 2)
    cmd0 = _build_pg_cmd(G_half, save_offset=0,      full_G=G)
    cmd1 = _build_pg_cmd(G_half, save_offset=G_half, full_G=G)
    gpus = [gpu_vae, gpu_tr]

    if dry_run:
        print(f"  [PG-parallel] G_half={G_half}")
        print(f"  CMD-GPU{gpu_vae}:", " ".join(cmd0))
        print(f"  CMD-GPU{gpu_tr}:", " ".join(cmd1))
        return {}

    env = os.environ.copy()
    env["TRANSFORMERS_VERBOSITY"] = "error"
    env["TOKENIZERS_PARALLELISM"] = "false"

    # ── Warmup: run G=1, steps=5 on each GPU (in parallel) ────────────────
    if do_warmup:
        wu0 = _build_pg_cmd(G_half, save_offset=0,      warmup=True, full_G=G)
        wu1 = _build_pg_cmd(G_half, save_offset=G_half, warmup=True, full_G=G)
        print(f"  [WARMUP] pg parallel (G_half={G_half} steps=3 on each GPU) ...", flush=True)
        wu_results: dict = {}
        def _wu(idx, cmd):
            e = env.copy()
            e["CUDA_VISIBLE_DEVICES"] = str(gpus[idx])
            p = subprocess.run(cmd, cwd=str(REPO), env=e, capture_output=True, text=True)
            wu_results[idx] = p
        wt0 = threading.Thread(target=_wu, args=(0, wu0))
        wt1 = threading.Thread(target=_wu, args=(1, wu1))
        wt0.start(); wt1.start(); wt0.join(); wt1.join()
        if all(wu_results[i].returncode == 0 for i in range(2)):
            print(f"  [WARMUP OK]", flush=True)
        else:
            print(f"  [WARMUP FAIL]", flush=True)

    # Reset peak-memory counters
    try:
        import torch
        for g in gpus:
            torch.cuda.reset_peak_memory_stats(g)
    except Exception:
        pass

    monitor = VRAMMonitor(gpus, interval=0.5)
    proc_results: dict = {}

    def _run(idx: int, cmd: list, gpu: int):
        e = env.copy()
        e["CUDA_VISIBLE_DEVICES"] = str(gpu)
        p = subprocess.run(cmd, cwd=str(REPO), env=e, capture_output=True, text=True)
        proc_results[idx] = p

    print(f"  [START] method=pg G={G}  GPUs=2 (G_half={G_half} per GPU)", flush=True)
    t0 = time.perf_counter()
    monitor.start()

    t0 = time.perf_counter()
    th0 = threading.Thread(target=_run, args=(0, cmd0, gpu_vae))
    th1 = threading.Thread(target=_run, args=(1, cmd1, gpu_tr))
    th0.start(); th1.start()
    th0.join(); th1.join()

    elapsed = time.perf_counter() - t0
    monitor.stop()

    ok = all(proc_results[i].returncode == 0 for i in range(2))
    if not ok:
        for i in range(2):
            p = proc_results[i]
            if p.returncode != 0:
                print(f"  [FAIL] GPU{gpus[i]} returncode={p.returncode}")
                tail = "\n".join(p.stderr.strip().splitlines()[-20:])
                print(f"  STDERR (tail):\n{tail}")

    peak = monitor.peak_mib
    print(f"  [DONE] elapsed={elapsed:.1f}s  peak_VRAM={peak}  ok={ok}", flush=True)

    return {
        "method":        "pg",
        "G":             G,
        "elapsed_s":     round(elapsed, 2),
        "ok":            ok,
        "peak_vram_mib": peak,
        "num_gpus":      2,
        "warmup_done":   do_warmup,
    }


# ─── Table printer ──────────────────────────────────────────────────────────

def _vram_str(row: dict) -> str:
    p = row.get("peak_vram_mib", {})
    if not p:
        return "—"
    parts = [f"GPU{g}={v}" for g, v in sorted(p.items())]
    return "|".join(parts) + " MiB"


def print_table(results: list[dict]):
    if not results:
        return
    methods = sorted({r["method"] for r in results})
    Gs      = sorted({r["G"]      for r in results})

    lookup = {}
    for r in results:
        lookup[(r["method"], r["G"])] = r

    col_w = 14
    header_parts = [f"{'Method':<18}  {'GPUs':<4}"]
    for G in Gs:
        header_parts.append(f"{'G='+str(G)+' time':>{col_w}}")
        header_parts.append(f"{'G='+str(G)+' VRAM':>{col_w+8}}")
    print("\n" + "  ".join(header_parts))
    print("  " + "-" * (24 + (col_w + col_w + 8 + 6) * len(Gs)))

    for m in methods:
        gpus_str = str(GPU_COUNT.get(m, "?"))
        row_parts = [f"{m:<18}  {gpus_str:<4}"]
        for G in Gs:
            r = lookup.get((m, G))
            if r is None:
                row_parts.append(f"{'—':>{col_w}}")
                row_parts.append(f"{'—':>{col_w+8}}")
            else:
                t = f"{r['elapsed_s']:.1f}s" if r["ok"] else "OOM/FAIL"
                row_parts.append(f"{t:>{col_w}}")
                row_parts.append(f"{_vram_str(r):>{col_w+8}}")
        print("  ".join(row_parts))


def save_results(results: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {path}")


# ─── Main ───────────────────────────────────────────────────────────────────

ALL_METHODS = ["base", "apg", "mix", "pg", "cads", "dpp", "ourmethod", "ourmethod_fast"]
ALL_G       = [4, 32]


def main():
    ap = argparse.ArgumentParser(description="Timing benchmark for all T2I methods")
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS,
                    choices=ALL_METHODS, metavar="METHOD",
                    help=f"Methods to run. ourmethod_fast uses --update-every 3. "
                         f"Choices: {ALL_METHODS}")
    ap.add_argument("--G", nargs="+", type=int, default=ALL_G,
                    help="Batch sizes to test (default: 4 32)")
    ap.add_argument("--warmup", action="store_true",
                    help="Run a G=same,steps=3 pass before timing each method. "
                         "Populates OS page cache AND compiles CuDNN kernels for the "
                         "target batch size, ensuring fair comparison across methods.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print commands without running")
    ap.add_argument("--gpu0", type=int, default=0,
                    help="GPU index for VAE / CLIP (default: 0)")
    ap.add_argument("--gpu1", type=int, default=1,
                    help="GPU index for transformer / text encoders (default: 1)")
    ap.add_argument("--out", type=str,
                    default=str(REPO / "results" / "benchmark_timing.json"),
                    help="Where to save JSON results")
    args = ap.parse_args()

    results = []
    total = len(args.methods) * len(args.G)
    done  = 0

    warmup_note = " (with per-method warmup)" if args.warmup else " (no warmup — beware OS cache effects)"
    gpu_vae, gpu_tr = args.gpu0, args.gpu1

    for G in sorted(args.G):
        for method in args.methods:
            done += 1
            print(f"\n[{done}/{total}] method={method}  G={G}  "
                  f"steps={STEPS}  {HEIGHT}×{WIDTH}  guidance={GUIDANCE}  "
                  f"GPUs=cuda:{gpu_vae}+cuda:{gpu_tr}")
            if method == "pg":
                r = run_pg_parallel(G, gpu_vae=gpu_vae, gpu_tr=gpu_tr,
                                    dry_run=args.dry_run, do_warmup=args.warmup)
            else:
                r = run_once(method, G, gpu_vae=gpu_vae, gpu_tr=gpu_tr,
                             dry_run=args.dry_run, do_warmup=args.warmup)
            if r:
                results.append(r)
                save_results(results, Path(args.out))  # save after every method

    if results:
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS" + warmup_note)
        print("=" * 80)
        print(f"Prompt   : {PROMPT!r}")
        print(f"Steps    : {STEPS}  |  Resolution: {HEIGHT}×{WIDTH}  |  Guidance: {GUIDANCE}")
        print(f"NOTE: All methods use 2 GPUs. pg runs as two parallel G//2 sub-processes "
              f"(FlowSampler is single-device; splitting ensures equal hardware usage).")
        print_table(results)
        save_results(results, Path(args.out))


if __name__ == "__main__":
    main()
