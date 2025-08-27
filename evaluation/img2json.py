#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert your generated image folders into the JSON formats expected by
facebookresearch/DIMCIM score calculators.

It supports two modes:
  1) DIM (coarse): one *flat* JSON list of absolute image paths
  2) CIM (dense): a JSON list of entries, each entry includes:
       {"prompt": str, "attribute_type": str, "attribute": str, "image_paths": [list-of-paths]}

You can provide multiple folders via a CSV manifest or via repeated --entry args.
- CSV columns: folder,prompt,attribute_type,attribute
- Repeated --entry example:
    --entry folder=/path/to/imgs, prompt="a beige table...", attribute_type=color, attribute=beige

Optionally, the script can try to *infer* attribute_type/attribute from folder names
like ".../color-beige/" using regex (see --infer_from_name).

Examples
--------
# (A) DIM (coarse) — flatten all images under the provided folders
python build_dimcim_json.py \
  --mode dim \
  --out ./dimcim_json/table_coarse_prompt_generated_images_paths.json \
  --folders ./runs/table/coarse_a ./runs/table/coarse_b

# (B) CIM (dense) — single prompt/folder
python build_dimcim_json.py \
  --mode cim \
  --out ./dimcim_json/table_dense_prompt_generated_images_paths.json \
  --entry folder=./runs/table/color-beige, prompt="A beige table in a room.", attribute_type=color, attribute=beige

# (C) CIM (dense) — many rows from CSV
# CSV (no header needed if you pass --no_header):
# folder,prompt,attribute_type,attribute
# ./runs/table/color-red,  "A red table by the window.",   color, red
# ./runs/table/color-blue, "A blue table in a studio.",    color, blue

python img2json.py --mode cim --out /mnt/data6t/yyz/flow_grpo/flow_base/outputs/ourMethod_An_airplane_wing_pointed_toward_the_ground/eval/cim.json --entry "folder=/mnt/data6t/yyz/flow_grpo/flow_base/outputs/ourMethod_An_airplane_wing_pointed_toward_the_ground/imgs, prompt=An airplane wing pointed toward the ground., attribute_type=color, attribute=airplane"

"""
from __future__ import annotations
import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import json
import re
from typing import List, Optional

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

def _list_images(folder: Path, recursive: bool = False) -> List[str]:
    if recursive:
        it = folder.rglob("*")
    else:
        it = folder.glob("*")
    paths = [str(p.resolve()) for p in it if p.suffix.lower() in IMG_EXTS]
    paths.sort()
    return paths

@dataclass
class DenseEntry:
    folder: Path
    prompt: str
    attribute_type: str
    attribute: str

# ---------------- CLI parsing helpers -----------------

def parse_entry_arg(s: str) -> DenseEntry:
    # format: folder=/path, prompt=..., attribute_type=..., attribute=...
    # allow commas inside prompt by accepting semicolons too; we first split on ',' not inside quotes.
    # Simple robust parser:
    kv = {}
    for part in re.split(r",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", s):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"Bad --entry segment: '{part}'. Expected key=value.")
        k, v = part.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"')
        kv[k] = v
    req = ["folder", "prompt", "attribute_type", "attribute"]
    missing = [k for k in req if k not in kv]
    if missing:
        raise ValueError(f"Missing keys in --entry: {missing}. Provide {', '.join(req)}")
    return DenseEntry(folder=Path(kv["folder"]).expanduser(), prompt=kv["prompt"],
                      attribute_type=kv["attribute_type"], attribute=kv["attribute"])

def read_manifest(csv_path: Path, no_header: bool=False) -> List[DenseEntry]:
    rows: List[DenseEntry] = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        if not no_header:
            header = next(reader)
            # normalize
            header = [h.strip().lower() for h in header]
            idx = {h:i for i,h in enumerate(header)}
            for required in ("folder","prompt","attribute_type","attribute"):
                if required not in idx:
                    raise ValueError(f"CSV must include column '{required}'. Got: {header}")
            for r in reader:
                rows.append(DenseEntry(
                    folder=Path(r[idx['folder']]).expanduser(),
                    prompt=r[idx['prompt']],
                    attribute_type=r[idx['attribute_type']],
                    attribute=r[idx['attribute']]
                ))
        else:
            for r in reader:
                if len(r) < 4:
                    raise ValueError("CSV row must be: folder,prompt,attribute_type,attribute")
                rows.append(DenseEntry(
                    folder=Path(r[0]).expanduser(),
                    prompt=r[1],
                    attribute_type=r[2],
                    attribute=r[3]
                ))
    return rows

# --------------- Main builders ------------------

def build_dim_json(folders: List[Path], recursive: bool=False) -> List[str]:
    all_paths: List[str] = []
    for d in folders:
        if not d.exists():
            raise FileNotFoundError(f"Folder not found: {d}")
        all_paths.extend(_list_images(d, recursive=recursive))
    # dedupe while preserving order
    seen = set()
    flat = []
    for p in all_paths:
        if p not in seen:
            seen.add(p)
            flat.append(p)
    return flat

def build_cim_json(entries: List[DenseEntry], recursive: bool=False) -> List[dict]:
    items = []
    for e in entries:
        if not e.folder.exists():
            raise FileNotFoundError(f"Folder not found: {e.folder}")
        paths = _list_images(e.folder, recursive=recursive)
        if len(paths) == 0:
            print(f"[WARN] No images found in {e.folder}")
        items.append({
            "prompt": e.prompt,
            "attribute_type": e.attribute_type,
            "attribute": e.attribute,
            "image_paths": paths,
        })
    return items

# --------------- Optional inference --------------

def infer_attr_from_name(folder: Path) -> Optional[tuple[str,str]]:
    """Try to infer (attribute_type, attribute) from folder path.
    Accepts patterns like '.../color-beige', '.../style_modern', '.../material-wood'.
    Returns (type, attr) or None.
    """
    name = folder.name.lower()
    m = re.search(r"(color|material|shape|style)[-_]([a-z0-9_]+)", name)
    if m:
        return m.group(1), m.group(2)
    return None

# --------------- CLI --------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dim","cim"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--folders", nargs="*", type=Path, help="Image folders (DIM mode) or fallback for CIM if used with --infer_from_name and --common_prompt")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders when listing images")

    # CIM options
    ap.add_argument("--manifest", type=Path, help="CSV with columns: folder,prompt,attribute_type,attribute")
    ap.add_argument("--no_header", action="store_true", help="Manifest CSV has no header row")
    ap.add_argument("--entry", action="append", default=[], help="Add one dense entry: folder=..., prompt=..., attribute_type=..., attribute=... (can repeat)")
    ap.add_argument("--infer_from_name", action="store_true", help="Try parsing attribute_type/attribute from folder name like 'color-beige'")
    ap.add_argument("--common_prompt", type=str, help="Use this prompt for all CIM entries when inferring from --folders")

    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "dim":
        if not args.folders:
            raise SystemExit("DIM mode requires --folders")
        flat = build_dim_json([Path(f).expanduser() for f in args.folders], recursive=args.recursive)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(flat, f, indent=2, ensure_ascii=False)
        print(f"[OK] Wrote DIM JSON with {len(flat)} paths -> {args.out}")
        return

    # CIM mode
    entries: List[DenseEntry] = []
    if args.manifest:
        entries.extend(read_manifest(args.manifest, no_header=args.no_header))
    for s in args.entry:
        entries.append(parse_entry_arg(s))

    if not entries and args.folders:
        if not args.common_prompt:
            raise SystemExit("When using --folders to infer CIM entries, also pass --common_prompt.")
        for d in args.folders:
            d = Path(d).expanduser()
            info = infer_attr_from_name(d) if args.infer_from_name else None
            if info is None:
                raise SystemExit(f"Cannot infer attribute from folder name: {d}. Provide --entry or --manifest, or enable --infer_from_name with a name like 'color-beige'.")
            atype, attr = info
            entries.append(DenseEntry(folder=d, prompt=args.common_prompt, attribute_type=atype, attribute=attr))

    if not entries:
        raise SystemExit("CIM mode needs at least one entry via --manifest or --entry or --folders + --infer_from_name + --common_prompt")

    data = build_cim_json(entries, recursive=args.recursive)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[OK] Wrote CIM JSON with {len(data)} prompt entries -> {args.out}")

if __name__ == "__main__":
    main()