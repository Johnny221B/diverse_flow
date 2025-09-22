# 运行示例：把本段保存为 run_and_aggregate.py 然后 python run_and_aggregate.py
import os
import sys
import subprocess
import json
import csv
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import argparse
import re

# ------------------------
# 用户需要修改 / 确认的项
# ------------------------
# evaluator_py: 你的 Unified Image Evaluation Tool 文件路径（包含 main() 的文件）
# 它应当支持命令行： --gen <folder> --real <real_dir> --out <out_json> --clip-jit <path> --device <device> ...
evaluator_py = "/mnt/data6t/yyz/flow_grpo/flow_base/evaluation/eval_imgs.py"  # <- 修改为你的 evaluator 文件名/路径

# 真实图像库（用于 FID/KID）
real_dir = "/mnt/data6t/yyz/flow_grpo/flow_base/datasets/COCO/dining_table"       # <- 修改为你的真实图像目录

# study_root: 指向某个 study 根目录，例如:
# <project_root>/outputs/add_robust_study/noise_gate
# 或
# <project_root>/outputs/add_robust_study/noise_scale
study_root = "/mnt/data6t/yyz/flow_grpo/flow_base/outputs/add_robust_study/noise_gate"  # <- 修改为待处理的 study 目录

# evaluator 运行参数（可根据需要调整）
clip_jit = os.path.expanduser("~/.cache/clip/ViT-B-32.pt")
device = "cuda:0"
batch_size = 32
num_workers = 4
clip_image_size = 224
eval_dir_name = "eval"  # 与 evaluator 一致

# ------------------------
# 内部工具函数
# ------------------------
def find_seed_folders(imgs_root: Path) -> List[Path]:
    """
    找到 imgs_root 下直接的每个 seed 子目录（不递归 deeper），
    也支持如果 imgs_root 里直接放图片则将 imgs_root 本身作为一个 folder。
    """
    if not imgs_root.exists():
        return []
    children = [p for p in sorted(imgs_root.iterdir())]
    seed_dirs = [p for p in children if p.is_dir()]
    if not seed_dirs:
        # imgs_root itself may be a single folder containing images
        return [imgs_root]
    return seed_dirs

def call_evaluator_for_folder(gen_folder: Path, real_folder: Path, out_json: Path) -> Tuple[int, str]:
    """
    调用 evaluator 脚本对单个生成文件夹做评价，并把结果写到 out_json。
    返回 (returncode, stdout+stderr)
    """
    cmd = [
        sys.executable, evaluator_py,
        "--gen", str(gen_folder),
        "--real", str(real_folder),
        "--out", str(out_json),
        "--device", device,
        "--batch-size", str(batch_size),
        "--num-workers", str(num_workers),
        "--clip-jit", str(clip_jit),
        "--clip-image-size", str(clip_image_size),
        "--eval_dir_name", eval_dir_name,
    ]
    # 记录命令以便调试
    print("Calling evaluator:", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    out = (res.stdout or "") + ("\nERR:\n" + res.stderr if res.stderr else "")
    return res.returncode, out

# ------------------------
# 聚合函数（读取 eval jsons 并输出 summary.csv）
# ------------------------
NUMERIC_FIELDS = [
    "num_images",
    "vendi_pixel",
    "vendi_inception",
    "fid",
    "kid_mean",
    "kid_std",
    "clip_score",
    "one_minus_ms_ssim",
    "brisque",
]

PATTERNS = {
    "noise_gate": re.compile(r"^seed\d+_noisegate(?P<t0>\d+(?:\.\d+)?)_(?P<t1>\d+(?:\.\d+)?)$"),
    "noise_scale": re.compile(r"^seed\d+_noisescale(?P<scale>\d+(?:\.\d+)?)$"),
}

def group_key_from_stem(study: str, stem: str) -> str:
    m = PATTERNS.get(study).match(stem) if study in PATTERNS else None
    if not m:
        return stem
    if study == "noise_gate":
        t0, t1 = m.group("t0"), m.group("t1")
        return f"noisegate{t0}_{t1}"
    else:
        scale = m.group("scale")
        return f"noisescale{scale}"

def is_number(x) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False

def agg_mean_std(vals: List[float]) -> Tuple[float, float]:
    if not vals:
        return (math.nan, math.nan)
    arr = np.array(vals, dtype=np.float64)
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else 0.0
    return mean, std

def aggregate_eval_jsons_to_csv(study: str, eval_dir: Path, out_csv_name: str = "summary.csv"):
    json_files = sorted([p for p in eval_dir.glob("*.json")])
    if not json_files:
        raise SystemExit(f"No JSON files found in {eval_dir}")

    groups = {}  # key -> list of metrics dicts
    for jf in json_files:
        stem = jf.stem  # e.g., seed1111_noisegate0.0_0.95
        key = group_key_from_stem(study, stem)
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Warning: failed to parse {jf}: {e}")
            continue
        results = data.get("results", {})
        if not results:
            print(f"Warning: empty results in {jf}")
            continue
        # get first value
        folder_name, metrics = next(iter(results.items()))
        groups.setdefault(key, []).append(metrics)

    rows = []
    for key in sorted(groups.keys()):
        metrics_list = groups[key]
        row = {"group": key, "num_seeds": len(metrics_list)}
        for field in NUMERIC_FIELDS:
            vals = []
            for m in metrics_list:
                v = m.get(field, None)
                if v is None:
                    continue
                if is_number(v):
                    vals.append(float(v))
            mean, std = agg_mean_std(vals)
            row[f"{field}_mean"] = mean
            row[f"{field}_std"] = std
        rows.append(row)

    header = ["group", "num_seeds"] + [f"{f}_mean" for f in NUMERIC_FIELDS] + [f"{f}_std" for f in NUMERIC_FIELDS]
    out_csv = eval_dir / out_csv_name
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Saved grouped CSV to: {out_csv}")

# ------------------------
# 主流程
# ------------------------
def main():
    parser = argparse.ArgumentParser(description="Batch-evaluate imgs/* folders (calls your evaluator) and aggregate into summary.csv")
    parser.add_argument("--study-root", type=str, default=study_root, help="study root path (e.g., .../noise_gate or .../noise_scale)")
    parser.add_argument("--real-dir", type=str, default=real_dir, help="real images directory")
    parser.add_argument("--evaluator-py", type=str, default=evaluator_py, help="path to your evaluator script")
    parser.add_argument("--device", type=str, default=device)
    parser.add_argument("--batch-size", type=int, default=batch_size)
    parser.add_argument("--num-workers", type=int, default=num_workers)
    parser.add_argument("--clip-jit", type=str, default=clip_jit)
    parser.add_argument("--clip-image-size", type=int, default=clip_image_size)
    args = parser.parse_args()

    imgs_root = Path(args.study_root) / "imgs"
    if not imgs_root.exists():
        raise SystemExit(f"Imgs root not found: {imgs_root}")

    eval_dir = Path(args.study_root) / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # update globals from args for the subprocess calls
    global evaluator_py, real_dir, device, batch_size, num_workers, clip_jit, clip_image_size
    evaluator_py = args.evaluator_py
    real_dir = args.real_dir
    device = args.device
    batch_size = args.batch_size
    num_workers = args.num_workers
    clip_jit = args.clip_jit
    clip_image_size = args.clip_image_size

    # find per-seed folders
    seed_folders = find_seed_folders(imgs_root)
    print(f"Found {len(seed_folders)} folders under {imgs_root} to evaluate.")

    # for each seed folder: call evaluator and write per-folder json to eval/
    for gen_folder in seed_folders:
        # name like seed1111_noisegate0.0_0.95 or seed1111_noisescale1.25
        stem = gen_folder.name
        out_json = eval_dir / f"{stem}.json"
        # skip if exists
        if out_json.exists():
            print(f"Skipping existing eval JSON: {out_json}")
            continue

        rc, out = call_evaluator_for_folder(gen_folder, Path(real_dir), out_json)
        if rc != 0:
            print(f"Evaluator failed for {gen_folder}. Return code {rc}. Output:\n{out}")
            # optionally continue or abort; here we continue
            continue
        else:
            print(f"Evaluator finished for {gen_folder}. Wrote {out_json}")

    # aggregate into CSV
    # determine study type
    study_name = Path(args.study_root).name  # noise_gate or noise_scale
    if study_name not in ("noise_gate", "noise_scale"):
        print(f"Warning: study root basename is '{study_name}', expected 'noise_gate' or 'noise_scale'. Using it as study id.")
    aggregate_eval_jsons_to_csv(study_name, eval_dir, out_csv_name="summary.csv")

if __name__ == "__main__":
    main()
