import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from pathlib import Path
import sys
import argparse
from typing import Dict, List, Tuple

# 尝试导入 scipy 进行平滑处理
try:
    from scipy.interpolate import PchipInterpolator
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARN] Scipy not found. Curves will remain jagged. Install with: pip install scipy", file=sys.stderr)

# -------------------------
# Font helpers (保留你的原始设置)
# -------------------------
def _pick_font(preferred_list):
    names = {f.name.lower(): f.name for f in fm.fontManager.ttflist}
    for want in preferred_list:
        for k, v in names.items():
            if want.lower() == k or want.lower() in k:
                return v
    return None

def _ensure_font(font_path=None, verbose=True):
    preferred = [
        "Times New Roman",
        "Times",
        "Nimbus Roman No9 L",
        "Liberation Serif",
        "DejaVu Serif",
    ]
    loaded_from_path = None
    if font_path:
        fpath = Path(font_path)
        if fpath.exists():
            try:
                fm.fontManager.addfont(str(fpath))
                prop = fm.FontProperties(fname=str(fpath))
                loaded_from_path = prop.get_name()
            except Exception as e:
                print(f"[WARN] failed to load font from --font-path: {e}", file=sys.stderr)

    picked = loaded_from_path or _pick_font(preferred) or mpl.rcParams.get("font.family", ["DejaVu Serif"])[0]
    mpl.rcParams["font.family"] = picked
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"]  = 42
    mpl.rcParams["axes.unicode_minus"] = False
    
    # 保留你增加的字体大小设置
    mpl.rcParams["font.size"] = 12        # 全局默认字体大小
    mpl.rcParams["axes.titlesize"] = 25   # 子图标题大小
    mpl.rcParams["axes.labelsize"] = 20   # 坐标轴标签大小
    mpl.rcParams["xtick.labelsize"] = 15  # x轴刻度标签大小
    mpl.rcParams["ytick.labelsize"] = 15  # y轴刻度标签大小
    mpl.rcParams["legend.fontsize"] = 15  # 图例字体大小
    
    if verbose:
        print(f"[INFO] Using font family: {picked}")

# -------------------------
# IO helpers (保持不变)
# -------------------------
def _try_csv_candidates(eval_dir: Path, concept: str, g_str: str):
    cand = []
    try:
        g_val = float(g_str)
        variants = list(dict.fromkeys([g_str, f"{g_val:g}", f"{g_val:.1f}", f"{g_val:.2f}"]))
    except ValueError: # Changed generic Exception to ValueError for specific error
        variants = [g_str]
    for v in variants:
        cand.append(eval_dir / f"exp1_{v}_{concept}.csv")
    for p in cand:
        if p.exists():
            return p
    for p in sorted(eval_dir.glob(f"exp1_*_{concept}.csv")):
        return p
    return None

def _load_one_csv(csv_path: Path, method: str, concept: str, guidance: float):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[ERR] Failed to read {csv_path}: {e}", file=sys.stderr)
        return None

    if "method" in df.columns:
        df = df[df["method"] == method]
    if "concept" in df.columns:
        df = df[df["concept"] == concept]
    if "guidance" in df.columns:
        df = df[(df["guidance"] - guidance).abs() < 1e-6]
    if df.empty:
        return None
    if "recall" in df.columns:
        df = df.sort_values(by=["recall"]).reset_index(drop=True)

    def _first(col): # Changed default=np.nan in original to make it robust
        return df[col].iloc[0] if col in df.columns and not df.empty else np.nan

    summary = {
        "auc_prd_mu": _first("auc_prd_mu"),
        "rec_at_p0.60_mu": _first("rec_at_p0.60_mu"),
        "rec_at_p0.70_mu": _first("rec_at_p0.70_mu"),
        "rec_at_p0.76_mu": _first("rec_at_p0.76_mu"),
    }
    return df, summary

def _resolve_sources(outputs_root: Path, methods_csv: str, files_csv: str, concept: str, labels_csv: str):
    files_or_roots, methods, labels = [], [], []
    
    if files_csv:
        toks = [t.strip() for t in files_csv.split(",") if t.strip()]
        files_or_roots = [Path(t) for t in toks]
        
        if methods_csv:
            m_toks = [m.strip() for m in methods_csv.split(",") if m.strip()]
            if len(m_toks) == len(files_or_roots):
                methods = m_toks
            else:
                print("[WARN] --methods count doesn't match --files, inferring from filenames...", file=sys.stderr)
                methods = []
        
        if not methods:
            for p in files_or_roots:
                try:
                    if "eval" in p.parts:
                        idx = p.parts.index("eval")
                        parent = p.parts[idx-1]
                        methods.append(parent.split("_")[0])
                    else: # Fallback for direct CSV path or non-standard structure
                        methods.append(p.stem.split('_')[0]) # e.g. exp1_3.0_concept.csv -> exp1
                except:
                    methods.append("Unknown")

        l_toks = [s.strip() for s in labels_csv.split(",")] if labels_csv else []
        labels = l_toks if len(l_toks) == len(files_or_roots) else methods[:]
        
    else:
        if not outputs_root or not methods_csv:
            print("[ERROR] Must provide (--outputs_root AND --methods) OR (--files)", file=sys.stderr)
            sys.exit(1)
        
        methods = [m.strip() for m in methods_csv.split(",") if m.strip()]
        files_or_roots = [outputs_root / f"{m}_{concept}" / "eval" for m in methods]
        
        l_toks = [s.strip() for s in labels_csv.split(",")] if labels_csv else []
        labels = l_toks if len(l_toks) == len(files_or_roots) else methods[:]

    return files_or_roots, methods, labels

# -------------------------
# Label normalization for legend (保持不变)
# -------------------------
def _legend_label(label: str):
    lower = label.strip().lower()
    if lower == "pg": return "PG"
    if lower == "dpp": return "DPP"
    if lower == "cads": return "CADS"
    if lower == "ourmethod": return "OSCAR"
    if lower == "oscar": return "OSCAR" # Added explicit 'oscar'
    return label

# -------------------------
# SMOOTHING LOGIC (核心平滑函数)
# -------------------------
def _smooth_curve(x: np.ndarray, y: np.ndarray, num_points: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用 PCHIP 插值将阶梯状/锯齿状数据转换为平滑曲线。
    处理重复的 X 值。
    """
    if not HAS_SCIPY or len(x) < 4: # PCHIP至少需要4个点
        # 如果Scipy不可用或点太少，返回原始数据
        sort_idx = np.argsort(x)
        return x[sort_idx], y[sort_idx]
    
    # 1. 排序
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    # 2. 处理重复的 X 值 (PCHIP要求x严格单调递增)。对于重复的 Recall 值，我们取 Precision 的均值。
    x_unique = []
    y_unique = []
    
    # 通过循环聚合重复的 x 值
    current_x = x_sorted[0]
    current_y_values = [y_sorted[0]]
    
    for i in range(1, len(x_sorted)):
        if x_sorted[i] == current_x:
            current_y_values.append(y_sorted[i])
        else:
            x_unique.append(current_x)
            y_unique.append(np.mean(current_y_values))
            current_x = x_sorted[i]
            current_y_values = [y_sorted[i]]
    
    # 添加最后一个点
    x_unique.append(current_x)
    y_unique.append(np.mean(current_y_values))

    x_unique = np.array(x_unique)
    y_unique = np.array(y_unique)
    
    if len(x_unique) < 4: # PCHIP至少需要4个点
        return x_unique, y_unique

    # 3. PCHIP 插值 (保形插值，避免 B-Spline 的过冲震荡)
    try:
        interpolator = PchipInterpolator(x_unique, y_unique)
        x_smooth = np.linspace(x_unique.min(), x_unique.max(), num_points)
        y_smooth = interpolator(x_smooth)
        
        # 截断到 [0, 1] 范围，防止插值超出物理限制
        y_smooth = np.clip(y_smooth, 0.0, 1.0)
        return x_smooth, y_smooth
    except Exception as e:
        print(f"[WARN] PCHIP smoothing failed: {e}. Returning raw data.", file=sys.stderr)
        return x_unique, y_unique


# -------------------------
# Plotting function (修改后的版本，严格遵守你的格式，只解决核心问题)
# -------------------------
def plot_guidances_panel(
    files_or_roots: List[Path],
    methods: List[str],
    labels: List[str],
    concept: str,
    guidances: List[str],
    save_dir: Path,
    title_suffix: str = "",
    xlim: Tuple[float, float] = None,
    ylim: Tuple[float, float] = None,
    legend_loc: str = "lower left",
):
    save_dir.mkdir(parents=True, exist_ok=True)
    Gs = [float(g) for g in guidances]
    n = len(Gs)

    base_colors = plt.get_cmap("tab10").colors
    color_map = {m: base_colors[i % len(base_colors)] for i, m in enumerate(methods)}

    fig, axes = plt.subplots(1, n, figsize=(3.3*n + 2.0, 3.2), sharex=True, sharey=True)
    axes = [axes] if n == 1 else list(axes)

    for ax_idx, ax in enumerate(axes):
        g = Gs[ax_idx]
        handles, leg_labels = [], []

        for i, src in enumerate(files_or_roots):
            method = methods[i]
            label  = labels[i] if labels and i < len(labels) else method

            # --- 1. 查找 CSV ---
            csv_path = src
            if src.is_dir():
                found_csv = _try_csv_candidates(src, concept, str(g))
                if found_csv is None:
                    print(f"[WARN] No CSV for method={method} (g={g}) under {src}", file=sys.stderr)
                    continue
                csv_path = found_csv
            
            # --- 2. 加载数据 ---
            loaded_data = _load_one_csv(csv_path, method, concept, float(g))
            if loaded_data is None:
                continue
            df, summary = loaded_data

            raw_x = df["recall"].values
            raw_y = df["precision_mu"].values
            # [恢复] 读取标准差
            raw_sd = df["precision_sd"].values if "precision_sd" in df.columns else np.zeros_like(raw_y)

            # --- 3. 数据平滑处理 (同时平滑均值和标准差) ---
            # 为了保证阴影和主线对齐，我们需要手动处理插值
            if HAS_SCIPY and len(raw_x) >= 4:
                # A. 排序
                sort_idx = np.argsort(raw_x)
                x_s = raw_x[sort_idx]
                y_s = raw_y[sort_idx]
                sd_s = raw_sd[sort_idx]

                # B. 处理重复x (取均值)
                x_uniq, y_uniq, sd_uniq = [], [], []
                curr_x = x_s[0]
                buf_y, buf_sd = [y_s[0]], [sd_s[0]]
                for k in range(1, len(x_s)):
                    if x_s[k] == curr_x:
                        buf_y.append(y_s[k])
                        buf_sd.append(sd_s[k])
                    else:
                        x_uniq.append(curr_x)
                        y_uniq.append(np.mean(buf_y))
                        sd_uniq.append(np.mean(buf_sd))
                        curr_x = x_s[k]
                        buf_y, buf_sd = [y_s[k]], [sd_s[k]]
                x_uniq.append(curr_x)
                y_uniq.append(np.mean(buf_y))
                sd_uniq.append(np.mean(buf_sd))
                
                x_uniq = np.array(x_uniq)
                y_uniq = np.array(y_uniq)
                sd_uniq = np.array(sd_uniq)

                if len(x_uniq) >= 4:
                    # C. PCHIP 插值
                    try:
                        x_plot = np.linspace(x_uniq.min(), x_uniq.max(), 500)
                        # 插值均值
                        interp_mu = PchipInterpolator(x_uniq, y_uniq)
                        y_plot = np.clip(interp_mu(x_plot), 0.0, 1.0)
                        # 插值标准差 (这样阴影也是平滑的)
                        interp_sd = PchipInterpolator(x_uniq, sd_uniq)
                        sd_plot = np.clip(interp_sd(x_plot), 0.0, 1.0)
                    except:
                        x_plot, y_plot, sd_plot = x_uniq, y_uniq, sd_uniq
                else:
                    x_plot, y_plot, sd_plot = x_uniq, y_uniq, sd_uniq
            else:
                # 不平滑的情况
                sort_idx = np.argsort(raw_x)
                x_plot = raw_x[sort_idx]
                y_plot = raw_y[sort_idx]
                sd_plot = raw_sd[sort_idx]

            # --- 4. 画图 ---
            # 主线
            ln, = ax.plot(x_plot, y_plot, lw=2.2, color=color_map[method])
            
            # [恢复] 阴影 (Fill Between)
            # alpha=0.15 表示透明度很低
            ax.fill_between(x_plot, 
                            np.clip(y_plot - sd_plot, 0, 1), 
                            np.clip(y_plot + sd_plot, 0, 1), 
                            color=color_map[method], 
                            alpha=0.15,  # <--- 这里控制透明度，越小越淡
                            lw=0)        # lw=0 去掉阴影边缘的细线

            handles.append(ln)
            leg_labels.append(f"{_legend_label(label)} (AUC={summary['auc_prd_mu']:.3f})")

        ax.set_yticks(np.arange(0.0, 1.0, 0.3))
        ax.grid(True, ls="--", alpha=0.35)
        if xlim: ax.set_xlim(*xlim)
        if ylim: ax.set_ylim(*ylim)
        ax.set_title(f"CFG={float(g):.1f}", fontsize=18)

        if handles:
            leg = ax.legend(handles, leg_labels, loc=legend_loc, frameon=True, fontsize=8)
            for lh in leg.legend_handles:
                lh.set_linewidth(2)

    fig.supxlabel("Recall", fontsize=15, y=0.08)
    fig.supylabel("Precision", fontsize=15, x=0.02)
    
    ttl = f"PRD in class-conditional task on Concept '{concept}'"
    if title_suffix:
        ttl += f" {title_suffix}"
    fig.suptitle(ttl, y=0.90, fontsize=20)
    
    fig.tight_layout(pad=1.0)

    gs_tag = "_".join([f"{float(g):g}" for g in Gs])
    out_png = save_dir / f"prd_{concept}_panel_cfg{gs_tag}.png"
    out_pdf = save_dir / f"prd_{concept}_panel_cfg{gs_tag}.pdf"
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"[SAVE] {out_png}")
    print(f"[SAVE] {out_pdf}")

    # if summary_rows:
    #     df_sum = pd.DataFrame(summary_rows)
    #     out_csv = save_dir / f"summary_{concept}_panel_cfg{gs_tag}.csv"
    #     df_sum.to_csv(out_csv, index=False)


# -------------------------
# CLI (修复后的版本，避免参数传递冲突)
# -------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs_root", type=Path, default=None,
                    help="like outputs/; expects {method}_{concept}/eval/exp1_<g>_<concept>.csv")
    ap.add_argument("--methods", type=str, default="",
                    help="comma-separated method names (prefix in {method}_{concept})")
    ap.add_argument("--files", type=str, default="",
                    help="comma-separated CSV paths or eval dirs; pairs with --labels")
    ap.add_argument("--labels", type=str, default="",
                    help="comma-separated legend labels; default uses method names")

    ap.add_argument("--concept", type=str, required=True)
    ap.add_argument("--guidances", type=str, required=True,
                    help="comma-separated, e.g. '3.0,5.0,7.5'")
    ap.add_argument("--save_dir", type=Path, default=None,
                    help="output dir; default is outputs_root/{concept}_exp1/plots or the parent of --files plus /plots")
    ap.add_argument("--xlim", type=str, default="", help="e.g. '0,0.25' (optional)")
    ap.add_argument("--ylim", type=str, default="", help="e.g. '0.6,1.0' (optional)")
    ap.add_argument("--title_suffix", type=str, default="", help="optional title suffix")
    ap.add_argument("--legend_loc", type=str, default="lower left",
                    help="legend location per subplot")
    ap.add_argument("--font-path", type=str, default="",
                    help="path to a .ttf for Times New Roman (optional)")

    args = ap.parse_args()
    _ensure_font(args.font_path, verbose=False)

    concept = args.concept
    guidances_list = [t.strip() for t in args.guidances.split(",") if t.strip()]

    # 1. 解析数据源
    files_or_roots_resolved, methods_resolved, labels_resolved = _resolve_sources(
        outputs_root=args.outputs_root,
        methods_csv=args.methods,
        files_csv=args.files,
        concept=concept,
        labels_csv=args.labels
    )

    # 2. 确定保存目录
    if args.save_dir is None:
        if args.outputs_root is not None:
            save_dir = args.outputs_root / f"{concept}_exp1" / "plots"
        else: # 尝试从 files_or_roots_resolved 推断
            base = files_or_roots_resolved[0] if files_or_roots_resolved else Path("./")
            save_dir = (base.parent if base.is_file() else base) / "plots"
    else:
        save_dir = args.save_dir

    # 3. 解析轴限制
    def parse_lim(s):
        if not s: return None
        try: return tuple(map(float, s.split(",")))
        except ValueError: return None # Changed generic Exception to ValueError

    xlim_parsed = parse_lim(args.xlim)
    ylim_parsed = parse_lim(args.ylim)

    # 4. 调用画图函数
    plot_guidances_panel(
        files_or_roots=files_or_roots_resolved,
        methods=methods_resolved,
        labels=labels_resolved,
        concept=concept,
        guidances=guidances_list,
        save_dir=save_dir,
        title_suffix=args.title_suffix,
        xlim=xlim_parsed,
        ylim=ylim_parsed,
        legend_loc=args.legend_loc,
    )