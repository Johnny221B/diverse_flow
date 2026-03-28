import argparse
import os
import shutil
import json
import csv
import subprocess
from pathlib import Path
from tqdm import tqdm

def create_symlink_dir(gen_imgs_root: Path, tmp_dir: Path):
    """
    将嵌套结构伪装成 T2I-CompBench 期望的扁平结构: tmp_dir/{prompt_text}_{idx}.png
    """
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    prompt_to_slug = {}

    for prompt_folder in gen_imgs_root.iterdir():
        if not prompt_folder.is_dir(): continue
        
        # 还原 Prompt 文本
        prompt_text = prompt_folder.name.replace("_", " ")
        prompt_to_slug[prompt_text] = prompt_folder.name

        img_files = list(prompt_folder.glob("*.png"))
        for idx, img_path in enumerate(img_files):
            # 兼容 T2I-CompBench 的解析逻辑: {prompt}_{index}.png
            symlink_name = f"{prompt_text}_{idx:04d}.png"
            symlink_path = tmp_dir / symlink_name
            try:
                os.symlink(img_path.absolute(), symlink_path)
            except OSError:
                # 如果系统不支持软链接，尝试复制（Windows 常见）
                shutil.copy(img_path.absolute(), symlink_path)
            
    return prompt_to_slug

def process_t2i_results(result_json_path: Path, eval_out_dir: Path, method: str, concept: str, prompt_to_slug: dict):
    if not result_json_path.exists():
        print(f"  [Error] 评分文件不存在: {result_json_path}")
        return

    with open(result_json_path, 'r') as f:
        results = json.load(f)

    score_dict = {}
    for item in results:
        prompt = item.get("prompt", "")
        score = item.get("score", 0.0)
        if prompt not in score_dict:
            score_dict[prompt] = []
        score_dict[prompt].append(score)

    csv_path = eval_out_dir / f"alignment_score_{concept}.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["prompt_folder", "prompt_text", f"{concept}_alignment_score"])
        
        for prompt, scores in score_dict.items():
            slug = prompt_to_slug.get(prompt, prompt.replace(" ", "_"))
            mean_score = sum(scores) / len(scores) if scores else 0.0
            writer.writerow([slug, prompt, round(mean_score, 4)])

    print(f"  [{method}] {concept} 结果回收成功 -> {csv_path}")

def run_t2i_evaluation(method: str, concept: str, outputs_root: Path, t2i_repo_path: Path, device: str):
    # 兼容两种文件夹命名
    method_dir = outputs_root / f"{method}_{concept}"
    if not (method_dir / "imgs").exists():
        method_dir = outputs_root / f"{method}_t2i_{concept}"
    
    if not (method_dir / "imgs").exists():
        print(f"[Skip] 找不到路径: {method_dir}/imgs")
        return

    gen_imgs_root = method_dir / "imgs"
    eval_out_dir = method_dir / "eval"
    eval_out_dir.mkdir(exist_ok=True)
    
    # 准备临时目录（放在 outputs 下）
    tmp_image_dir = outputs_root / "tmp_t2i_eval_images"
    prompt_to_slug = create_symlink_dir(gen_imgs_root, tmp_image_dir)

    print(f"\n>>> 正在评价: {method} | Concept: {concept}")

    original_cwd = os.getcwd()
    
    try:
        if concept == "color":
            # 必须在 BLIPvqa_eval 目录下运行以识别内部路径
            task_dir = t2i_repo_path / "BLIPvqa_eval"
            os.chdir(task_dir)
            
            # 添加了 --out_dir 参数，指定存到当前目录
            cmd = f"python BLIP_vqa.py --dataset color --image_dir {tmp_image_dir.absolute()} --device {device} --out_dir ."
            print(f"  [Exec] {cmd}")
            subprocess.run(cmd, shell=True, check=True)
            
            result_json = task_dir / "vqa_result.json"
            process_t2i_results(result_json, eval_out_dir.absolute(), method, concept, prompt_to_slug)
            
        elif concept == "spatial":
            # 必须在 UniDet_eval 目录下运行以读取 dataset/ 文件夹
            task_dir = t2i_repo_path / "UniDet_eval"
            os.chdir(task_dir)
            
            cmd = f"python 2D_spatial_eval.py --image_dir {tmp_image_dir.absolute()}"
            print(f"  [Exec] {cmd}")
            subprocess.run(cmd, shell=True, check=True)
            
            result_json = task_dir / "spatial_result.json"
            process_t2i_results(result_json, eval_out_dir.absolute(), method, concept, prompt_to_slug)
            
        elif concept == "complex":
            task_dir = t2i_repo_path / "3_in_1_eval"
            os.chdir(task_dir)
            
            cmd = f"python 3_in_1.py --image_dir {tmp_image_dir.absolute()}"
            print(f"  [Exec] {cmd}")
            subprocess.run(cmd, shell=True, check=True)
            
            result_json = task_dir / "complex_result.json"
            process_t2i_results(result_json, eval_out_dir.absolute(), method, concept, prompt_to_slug)
            
    except Exception as e:
        print(f"  [Error] 运行失败: {e}")
    finally:
        os.chdir(original_cwd)
        if tmp_image_dir.exists():
            shutil.rmtree(tmp_image_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", type=str, default=os.path.abspath("./outputs"))
    parser.add_argument("--t2i-repo-path", type=str, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--concepts", nargs="+", default=["color", "spatial", "complex"])
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    t2i_repo_path = Path(args.t2i_repo_path)

    for method in args.methods:
        for concept in args.concepts:
            run_t2i_evaluation(method, concept, outputs_root, t2i_repo_path, args.device)

if __name__ == "__main__":
    main()