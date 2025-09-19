#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# plot_cov_tau.py (edited)

import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["axes.unicode_minus"] = False

try:
    from scipy.interpolate import PchipInterpolator as _PCHIP
    _HAS_PCHIP = True
except Exception:
    _PCHIP = None
    _HAS_PCHIP = False


def _guidance_name_candidates(g):
    cands = []
    if isinstance(g, str):
        try:
            gv = float(g)
        except ValueError:
            return [g]
    else:
        gv = float(g)
    cands.append(str(g))
    for dp in (1, 2, 3, 4):
        cands.append(f"{gv:.{dp}f}")
    if math.isclose(gv, round(gv), rel_tol=0, abs_tol=1e-9):
        cands.append(str(int(round(gv))))
    seen, out = set(), []
    for s in cands:
        if s not in seen:
            out.append(s); seen.add(s)
    return out


def _build_csv_path(outputs_root: Path, method: str, concept: str, guidance) -> Path | None:
    mc_eval = outputs_root / f"{method}_{concept}" / "eval"
    for gname in _guidance_name_candidates(guidance):
        cand = mc_eval / f"exp3_{gname}_{concept}.csv"
        if cand.is_file():
            return cand
    return None


def _read_one_guidance_df(outputs_root: Path, concept: str, methods: list[str], guidance) -> pd.DataFrame:
    csv_paths = []
    for m in methods:
        p = _build_csv_path(outputs_root, m, concept, guidance)
        if p is not None:
            csv_paths.append((m, p))
    if not csv_paths:
        raise FileNotFoundError("No CSV found for any method at this guidance.")

    frames = []
    for m, p in csv_paths:
        df = pd.read_csv(p)
        df = df[(df["concept"].astype(str).str.lower() == concept.lower()) &
                (df["guidance"].astype(float) == float(guidance))]
        if df.empty:
            continue
        df["__src__"] = str(p)
        frames.append(df)

    if not frames:
        raise ValueError("Data empty after filtering by concept/guidance.")

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["method", "tau"]).drop_duplicates(["method","tau","guidance","concept"])
    return df


def _smooth_xy(x, y, x_dense):
    x = np.asarray(x, float); y = np.asarray(y, float); xd = np.asarray(x_dense, float)
    uniq_x, idx = np.unique(x, return_index=True)
    uniq_y = y[idx]
    order = np.argsort(uniq_x)
    ux, uy = uniq_x[order], uniq_y[order]
    yd = np.interp(xd, ux, uy)
    if _HAS_PCHIP and ux.size >= 2:
        try:
            f = _PCHIP(ux, uy, extrapolate=False)
            y2 = f(xd)
            mask = (xd >= ux[0]) & (xd <= ux[-1])
            yd[mask] = y2[mask]
        except Exception:
            pass
    yd[xd < ux[0]] = uy[0]
    yd[xd > ux[-1]] = uy[-1]
    return yd


def _norm_label(name: str) -> str:
    lo = name.strip().lower()
    if lo == "pg": return "PG"
    if lo == "dpp": return "DPP"
    if lo == "cads": return "CADS"
    if lo == "ourmethod": return "Ourmethod"
    return name


def plot_cov_tau_panel(
    outputs_root: str | Path,
    concept: str,
    methods: list[str],
    guidances: list[str],
    save_path: str | None = None,
    title: str | None = None,
    show_std: bool = True,
    tau_min: float = 0.05,
    tau_max: float = 0.65,
    dense: int = 200,
    legend_loc: str = "upper right",
    show_points: bool = True,
):
    outputs_root = Path(outputs_root)

    colors = plt.get_cmap("tab10").colors
    color_map = {m: colors[i % len(colors)] for i, m in enumerate(methods)}

    n = len(guidances)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n + 1.8, 3.2), sharey=True, sharex=True)
    if n == 1:
        axes = [axes]
    mid_idx = n // 2

    plotted_methods = set()

    for idx, (ax, g) in enumerate(zip(axes, guidances)):
        try:
            df = _read_one_guidance_df(outputs_root, concept, methods, g)
        except (FileNotFoundError, ValueError) as e:
            ax.set_title(f"CFG={float(g):.1f}\n(no data)", fontsize=11)
            ax.set_xlim(tau_min, tau_max)
            ax.set_ylim(0.0, 0.6)
            ax.grid(True, alpha=0.3, ls="--")
            continue

        df = df[(df["tau"] >= tau_min) & (df["tau"] <= tau_max)]
        x_dense = np.linspace(tau_min, tau_max, dense)

        if df.empty:
            ax.set_title(f"CFG={float(g):.1f}\n(no data)", fontsize=11)
        else:
            for m in methods:
                sub = df[df["method"] == m].sort_values("tau")
                if sub.empty:
                    continue
                x = sub["tau"].astype(float).values
                y = sub["coverage_mean"].astype(float).values
                y_line = _smooth_xy(x, y, x_dense)

                ax.plot(x_dense, y_line, lw=2.2, color=color_map[m], label=_norm_label(m))
                if show_points:
                    ax.plot(x, y, "o", ms=3.5, color=color_map[m])

                if show_std and "coverage_std" in sub.columns:
                    sd = sub["coverage_std"].astype(float).values
                    sd_line = _smooth_xy(x, sd, x_dense)
                    y_lo = np.clip(y_line - sd_line, 0.0, 1.0)
                    y_hi = np.clip(y_line + sd_line, 0.0, 1.0)
                    ax.fill_between(x_dense, y_lo, y_hi, color=color_map[m], alpha=0.15, lw=0)

                plotted_methods.add(m)

        ax.set_xlim(tau_min, tau_max)
        ax.set_ylim(0.0, 0.6)  # <-- y-limit as requested
        ax.grid(True, alpha=0.3, ls="--")
        ax.set_title(f"CFG={float(g):.1f}", fontsize=11)

        if idx == mid_idx:
            ax.set_xlabel(r"Threshold $\tau$")
        else:
            ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=True, bottom=True)

    axes[0].set_ylabel("Coverage")

    legend_methods = [m for m in methods if m in plotted_methods] or methods
    fig.legend(
        handles=[plt.Line2D([0], [0], color=color_map[m], lw=2.2) for m in legend_methods],
        labels=[_norm_label(m) for m in legend_methods],
        loc=legend_loc,
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.98, 0.98),
        bbox_transform=fig.transFigure,
    )

    if title is None:
        title = f"Mode Coverage vs. Threshold ($\\tau$) for the '{concept}' Concept"
    fig.suptitle(title, y=0.95, fontsize=14)

    fig.tight_layout(rect=[0, 0, 0.9, 1])

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[SAVE] {save_path}")
        pdf = Path(save_path).with_suffix(".pdf")
        fig.savefig(pdf, bbox_inches="tight")
        print(f"[SAVE] {pdf}")
        plt.close(fig)
    else:
        plt.show()


def _parse_list(s: str) -> list[str]:
    return [t.strip() for t in s.split(",") if t.strip()]


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", required=True, type=str)
    ap.add_argument("--methods", required=True, type=str, help="comma-separated, e.g., dpp,pg,cads")
    ap.add_argument("--guidances", required=True, type=str, help="comma-separated, e.g., 3.0,5.0,7.5")
    ap.add_argument("--outputs_root", type=str, default="/mnt/data/flow_grpo/flow_base/outputs")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--title", type=str, default=None)
    ap.add_argument("--no_std", action="store_true")
    ap.add_argument("--tau_min", type=float, default=0.05)
    ap.add_argument("--tau_max", type=float, default=0.65)
    ap.add_argument("--dense", type=int, default=200)
    ap.add_argument("--legend_loc", type=str, default="upper right")
    ap.add_argument("--no_points", action="store_true")
    args = ap.parse_args()

    methods = _parse_list(args.methods)
    guidances = _parse_list(args.guidances)

    plot_cov_tau_panel(
        outputs_root=args.outputs_root,
        concept=args.concept,
        methods=methods,
        guidances=guidances,
        save_path=args.out,
        title=args.title,
        show_std=not args.no_std,
        tau_min=args.tau_min,
        tau_max=args.tau_max,
        dense=args.dense,
        legend_loc=args.legend_loc,
        show_points=not args.no_points,
    )


if __name__ == "__main__":
    _main()
