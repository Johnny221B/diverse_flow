#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# ---------- I/O ----------
def read_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

# ---------- colors & order ----------
def darken_color(color, amount=0.35):
    try:
        c = mcolors.cnames[color]
    except Exception:
        c = color
    r, g, b = mcolors.to_rgb(c)
    return [(1 - amount) * r, (1 - amount) * g, (1 - amount) * b]

def lighten_color(color, amount=0.85):
    r, g, b = mcolors.to_rgb(color)
    return [1 - (1 - r) * (1 - amount), 1 - (1 - g) * (1 - amount), 1 - (1 - b) * (1 - amount)]

def collect_order_union(method_to_data: dict):
    methods = list(method_to_data.keys())
    first = method_to_data[methods[0]]
    order, by_type = [], {}
    for t, inner in first.items():
        by_type.setdefault(t, [])
        for a in inner.keys():
            by_type[t].append(a)
            order.append((t, a))
    for m in methods[1:]:
        for t, inner in method_to_data[m].items():
            by_type.setdefault(t, [])
            for a in inner.keys():
                if a not in by_type[t]:
                    by_type[t].append(a)
                    order.append((t, a))
    type_order = list(first.keys()) + [t for t in by_type if t not in first]
    order2 = []
    for t in type_order:
        for a in by_type[t]:
            order2.append((t, a))
    return order2, type_order, by_type

# ---------- style patterns ----------
STYLE_LIBRARY = {
    "solid":       dict(hatch=None,    face_mode="solid"),
    "dense_grid":  dict(hatch="xxxx",  face_mode="light"),
    "diag":        dict(hatch="////",  face_mode="light"),
    "backslash":   dict(hatch="\\\\\\\\", face_mode="light"),
    "dots":        dict(hatch="..",    face_mode="light"),
    "rings":        dict(hatch="OO",    face_mode="light"),
    "cross":        dict(hatch="++",    face_mode="light"),
    "bars":        dict(hatch="|||",   face_mode="light"),
}
DEFAULT_STYLE_CYCLE = ["solid", "dense_grid", "diag", "dots", "backslash", "rings", "cross", "bars"]

def parse_style_map(s: str | None) -> dict:
    if not s: return {}
    out = {}
    for kv in s.split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k.strip()] = v.strip()
    return out

# ---------- two-line labels ----------
def _two_lines(label: str, min_len: int = 14) -> str:
    if len(label) < min_len or "-" not in label:
        return label
    parts = label.split("-")
    best = None
    for k in range(1, len(parts)):
        left = "-".join(parts[:k])
        right = "-".join(parts[k:])
        score = abs(len(left) - len(right))
        if best is None or score < best[0]:
            best = (score, f"{left}\n{right}")
    return best[1]

# ---------- method label formatting ----------
def _format_method_label(method: str) -> str:
    method_lower = method.strip().lower()
    if method_lower == "pg":
        return "PG"
    elif method_lower == "dpp":
        return "DPP"
    elif method_lower == "cads":
        return "CADS"
    elif method_lower == "ourmethod":
        return "Ourmethod"
    return method

# ---------- plotting ----------
def plot_scores_multi(method_to_data: dict, score_type="DIM", save_path="plot.png", method_styles: dict | None=None):
    order, type_order, attrs_by_type = collect_order_union(method_to_data)
    cmap = plt.get_cmap('tab10')
    type_colors  = {t: cmap(i % 10) for i, t in enumerate(type_order)}
    bar_base_col = {t: darken_color(type_colors[t], amount=0.35) for t in type_order}

    raw_labels = [f"{t}-{a}" for (t, a) in order]
    labels = [_two_lines(s) for s in raw_labels]
    x = np.arange(len(labels))
    methods = list(method_to_data.keys())
    M = len(methods)
    bar_width = min(0.82 / M, 0.28)
    offsets = [(i - (M-1)/2) * bar_width for i in range(M)]

    fig, ax = plt.subplots(figsize=(12, 6))

    if score_type.upper() == "DIM":
        y_min, y_max = -0.40, 0.60
        legend_loc = 'lower right'
    elif score_type.upper() == "CIM":
        y_min, y_max = 0.00, 1.00
        legend_loc = 'upper right'
    else:
        y_min, y_max = -1, 1
        legend_loc = 'lower right'

    start = 0
    for t in type_order:
        n = len(attrs_by_type[t])
        end = start + n
        ax.axvspan(start - 0.5, end - 0.5, color=type_colors[t], alpha=0.18)
        start = end

    handles = []
    for mi, method in enumerate(methods):
        style_name = (method_styles or {}).get(method, DEFAULT_STYLE_CYCLE[mi % len(DEFAULT_STYLE_CYCLE)])
        style = STYLE_LIBRARY.get(style_name, STYLE_LIBRARY["solid"])
        hatch = style["hatch"]; face_mode = style["face_mode"]

        vals, colors = [], []
        for (t, a) in order:
            vals.append(method_to_data.get(method, {}).get(t, {}).get(a, np.nan))
            colors.append(bar_base_col[t] if face_mode == "solid" else lighten_color(type_colors[t], 0.9))
        vals = np.asarray(vals, dtype=float)
        x_i = x + offsets[mi]

        for j, v in enumerate(vals):
            if np.isnan(v): continue
            ax.bar(x_i[j], v, width=bar_width, color=colors[j],
                   edgecolor="black", linewidth=0.9, hatch=hatch)

        legend_face = "white" if face_mode != "solid" else bar_base_col[type_order[0]]
        formatted_label = _format_method_label(method)
        handles.append(mpatches.Patch(facecolor=legend_face, edgecolor="black", hatch=hatch, label=formatted_label))

    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.8)
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel(score_type)
    ax.set_xlabel('Attributes')
    ax.set_title(f'{score_type}: multi-method comparison')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha='center', linespacing=0.9)
    ax.tick_params(axis='x', labelsize=8)

    for tick_label, c in zip(ax.get_xticklabels(), [type_colors[t] for (t, a) in order]):
        tick_label.set_color(c)

    ax.legend(handles=handles, title="Method", ncol=min(len(methods), 4), frameon=True, loc=legend_loc)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=220, bbox_inches='tight')
    plt.close(fig)

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Compare DIM/CIM across methods and save a single plot.")
    ap.add_argument("--root", default=".", help="Root path containing outputs/")
    ap.add_argument("--methods", nargs="+", required=True, help="Method list, e.g., pg cads dpp")
    ap.add_argument("--concept", required=True, help="Concept, e.g., bus")
    ap.add_argument("--score-type", choices=["DIM", "CIM"], default="DIM", help="Choose to plot DIM or CIM")
    ap.add_argument("--styles", default=None,
                    help="Optional: method to style mapping, e.g., 'pg=solid,cads=dense_grid,dpp=diag'")
    ap.add_argument("--save-dir", default="./experiment4", help="Save directory (default ./experiment4)")
    args = ap.parse_args()

    scores_file = "dim_scores.json" if args.score_type.upper() == "DIM" else "cim_scores.json"

    method_to_data = {}
    missing = []
    for m in args.methods:
        p = Path(args.root) / "outputs" / f"{m}_{args.concept}_CIMDIM" / "eval" / scores_file
        if not p.exists():
            missing.append(str(p))
            continue
        method_to_data[m] = read_json(p)

    if not method_to_data:
        raise FileNotFoundError(f"No available {scores_file} found:\n" + "\n".join(missing))

    save_path = Path(args.save_dir) / f"{args.concept}_{args.score_type.upper()}.png"
    method_styles = parse_style_map(args.styles)
    plot_scores_multi(method_to_data, score_type=args.score_type.upper(), save_path=str(save_path), method_styles=method_styles)
    print(f"[OK] saved: {save_path.resolve()}")

if __name__ == "__main__":
    main()