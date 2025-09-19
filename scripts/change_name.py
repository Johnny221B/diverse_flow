#!/usr/bin/env python3
# rename_dirs.py
import argparse
from pathlib import Path

def rename_dirs(root: Path, recursive: bool, dry_run: bool):
    # 选择遍历顺序：先处理更深层，避免父目录改名影响子目录路径
    it = root.rglob("*") if recursive else root.iterdir()
    # 只保留目录，并按路径深度从深到浅排序
    dirs = [p for p in it if p.is_dir()]
    dirs.sort(key=lambda p: len(p.parts), reverse=True)

    changed = 0
    for d in dirs:
        if "potted_plant" not in d.name:
            continue
        new_name = d.name.replace("potted_plant", "bus")
        target = d.with_name(new_name)

        if target.exists():
            print(f"[跳过] 目标已存在：{d} -> {target}")
            continue

        if dry_run:
            print(f"[模拟] {d} -> {target}")
        else:
            d.rename(target)
            print(f"[已改]  {d} -> {target}")
            changed += 1
    if dry_run:
        print(f"\n模拟完成：将会改名 {sum('potted_plant' in p.name for p in dirs)} 个目录。")
    else:
        print(f"\n完成：实际改名 {changed} 个目录。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="将目录名中的 'potted_plant' 替换为 'bus'"
    )
    parser.add_argument("root", type=Path, help="要处理的根目录路径")
    parser.add_argument(
        "-r", "--recursive", action="store_true",
        help="递归处理所有子目录（默认只处理第一层）"
    )
    parser.add_argument(
        "-n", "--dry-run", action="store_true",
        help="仅显示将进行的改名，不实际修改"
    )
    args = parser.parse_args()

    if not args.root.exists() or not args.root.is_dir():
        raise SystemExit(f"路径不存在或不是目录：{args.root}")

    rename_dirs(args.root, args.recursive, args.dry_run)