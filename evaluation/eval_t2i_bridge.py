import argparse
import os
import shutil
import json
import csv
import subprocess
import time
from pathlib import Path
from tqdm import tqdm

def prepare_benchmark_samples(gen_imgs_root: Path, benchmark_samples_dir: Path):
    """
    将嵌套结构伪装成 T2I-CompBench 硬编码期望的扁平结构
    格式: benchmark_samples_dir/{prompt}_{index:06d}.png
    """
    if benchmark_samples_dir.exists():
        shutil.rmtree(benchmark_samples_dir)
    benchmark_samples_dir.mkdir(parents=True)

    _log(f"正在准备图片至: {benchmark_samples_dir}")
    prompt_to_slug = {}

    # 遍历每个 prompt 文件夹
    for prompt_folder in gen_imgs_root.iterdir():
        if not prompt_folder.is_dir(): continue
        
        prompt_text = prompt_folder.name.replace("_", " ")
        prompt_to_slug[prompt_text] = prompt_folder.name

        img_files = sorted(list(prompt_folder.glob("*.png")))
        for idx, img_path in enumerate(img_files):
            # T2I-CompBench 内部解析逻辑通常是 split('_')[-1]
            # 必须保证文件名是 prompt_序号.png
            target_name = f"{prompt_text}_{idx:06d}.png"
            target_path = benchmark_samples_dir / target_name
            
            # 使用硬链接最快且不占额外空间
            try:
                os.link(img_path, target_path)
            except:
                shutil.copy(img_path, target_path)
            
    return prompt_to_slug

def process_and_save_csv(json_path: Path, csv_save_path: Path, prompt_to_slug: dict, concept: str):
    """解析 T2I 输出的 JSON 并按 Prompt 汇总到 CSV"""
    if not json_path.exists():
        _log(f"错误: 找不到结果文件 {json_path}", is_error=True)
        return

    with open(json_path, 'r') as f:
        results = json.load(f)

    score_map = {}
    for item in results:
        p = item.get("prompt", "")
        s = item.get("score", 0.0)
        if p not in score_map: score_map[p] = []
        score_map[p].append(s)

    with open(csv_save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["prompt_folder", "prompt_text", f"{concept}_score"])
        for p, scores in score_map.items():
            slug = prompt_to_slug.get(p, p.replace(" ", "_"))
            writer.writerow([slug, p, round(sum(scores)/len(scores), 4)])

def _log(msg, is_error=False):
    prefix = "[ERROR]" if is_error else "[INFO]"
    print(f"{prefix} {time.strftime('%H:%M:%S')} {msg}", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", type=str, required=True, help="你的 outputs 根目录")
    parser.add_argument("--t2i-root", type=str, required=True, help="T2I-CompBench 仓库根目录")
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--concepts", nargs="+", default=["color", "spatial", "complex"])
    args = parser.parse_args()

    out_root = Path(args.outputs_root)
    t2i_root = Path(args.t2i_root)
    # T2I-CompBench 脚本默认寻找的图片路径
    samples_bridge_dir = t2i_root / "examples" / "samples"

    for method in args.methods:
        for concept in args.concepts:
            # 1. 路径定位
            method_dir = out_root / f"{method}_{concept}"
            if not method_dir.exists():
                method_dir = out_root / f"{method}_t2i_{concept}"
            
            if not (method_dir / "imgs").exists():
                _log(f"跳过: 未找到目录 {method_dir}", is_error=True)
                continue

            _log(f"开始测试: Method={method}, Concept={concept}")

            # 2. 转换数据 (Flattening)
            prompt_to_slug = prepare_benchmark_samples(method_dir / "imgs", samples_bridge_dir)

            # 3. 切换目录并执行 (调度对应脚本)
            cwd = os.getcwd()
            try:
                if concept == "color":
                    os.chdir(t2i_root / "BLIPvqa_eval")
                    # 仅使用脚本支持的参数
                    cmd = "python BLIP_vqa.py --out_dir ."
                    subprocess.run(cmd, shell=True, check=True)
                    process_and_save_csv(Path("vqa_result.json"), method_dir / "eval" / "alignment_color.csv", prompt_to_slug, "color")

                elif concept == "spatial":
                    os.chdir(t2i_root / "UniDet_eval")
                    # 该脚本默认输出到当前目录
                    cmd = "python 2D_spatial_eval.py"
                    subprocess.run(cmd, shell=True, check=True)
                    process_and_save_csv(Path("spatial_result.json"), method_dir / "eval" / "alignment_spatial.csv", prompt_to_slug, "spatial")

                elif concept == "complex":
                    os.chdir(t2i_root / "3_in_1_eval")
                    # 该脚本强制要求 --outpath
                    cmd = "python 3_in_1.py --outpath complex_result.json"
                    subprocess.run(cmd, shell=True, check=True)
                    process_and_save_csv(Path("complex_result.json"), method_dir / "eval" / "alignment_complex.csv", prompt_to_slug, "complex")

            except Exception as e:
                _log(f"运行失败: {e}", is_error=True)
            finally:
                os.chdir(cwd)

    # 清理桥接目录
    if samples_bridge_dir.exists():
        shutil.rmtree(samples_bridge_dir)
    _log("所有测试任务已完成。")

if __name__ == "__main__":
    main()