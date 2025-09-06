# plot_cov_tau.py
import argparse
from pathlib import Path
import math
import pandas as pd
import matplotlib.pyplot as plt

def _guidance_name_candidates(g):
    """
    尝试多种字符串形式以匹配不同保存方式：
    例如 3.0 -> ["3.0", "3", "3.00", "3.000", "3.0000"]
    """
    cands = []
    if isinstance(g, str):
        try:
            gv = float(g)
        except ValueError:
            # 字符串直接用
            return [g]
    else:
        gv = float(g)

    # 原样
    cands.append(str(g))
    # 1~4 位小数
    for dp in (1, 2, 3, 4):
        cands.append(f"{gv:.{dp}f}")
    # 纯整数（如果接近整数）
    if math.isclose(gv, round(gv), rel_tol=0, abs_tol=1e-9):
        cands.append(str(int(round(gv))))
    # 去重保序
    seen, out = set(), []
    for s in cands:
        if s not in seen:
            out.append(s); seen.add(s)
    return out

def _build_csv_path(outputs_root: Path, method: str, concept: str, guidance) -> Path | None:
    """
    根据 root/method_concept/eval/exp3_guidance_concept.csv 规则，尝试多种 guidance 命名。
    找到存在的文件就返回；找不到返回 None。
    """
    mc_root = outputs_root / f"{method}_{concept}"
    eval_dir = mc_root / "eval"
    for gname in _guidance_name_candidates(guidance):
        cand = eval_dir / f"exp3_{gname}_{concept}.csv"
        if cand.is_file():
            return cand
    return None

def plot_cov_tau(
    outputs_root: str | Path,
    concept: str,
    methods: list[str],
    guidance,
    title: str | None = None,
    save_path: str | None = None,
    show_std: bool = True,
    method_order: list[str] | None = None,
):
    """
    自动读取多方法的 exp3_{guidance}_{concept}.csv 并绘制 Coverage–τ 曲线。
    期望列：method, concept, guidance, k, tau, coverage_mean, coverage_std
    """
    outputs_root = Path(outputs_root)
    csv_paths = []
    for m in methods:
        p = _build_csv_path(outputs_root, m, concept, guidance)
        if p is None:
            print(f"[WARN] 找不到 CSV：{outputs_root}/{m}_{concept}/eval/exp3_*_{concept}.csv (guidance={guidance}) — 跳过 {m}")
        else:
            csv_paths.append((m, p))

    if not csv_paths:
        raise FileNotFoundError("一个 CSV 都没找到。请检查 outputs_root / concept / methods / guidance 是否匹配。")

    # 读入并合并
    frames = []
    for m, p in csv_paths:
        df = pd.read_csv(p)
        # 只保留与 concept/guidance 匹配的数据，防止误读到别的
        df = df[(df["concept"].astype(str).str.lower() == concept.lower()) &
                (df["guidance"].astype(float) == float(guidance))]
        if df.empty:
            print(f"[WARN] 文件存在但过滤后为空：{p}")
            continue
        df["__src__"] = str(p)
        frames.append(df)

    if not frames:
        raise ValueError("过滤后没有数据（检查 CSV 内容中的 concept/guidance 列）。")

    df = pd.concat(frames, ignore_index=True)
    # 排序去重
    df = df.sort_values(["method", "tau"]).drop_duplicates(["method","tau","guidance","concept"])

    # 方法顺序：默认按传入 methods 的顺序
    if method_order:
        methods_plot = [m for m in method_order if m in set(df["method"])]
    else:
        # 用调用方传入的 methods 顺序（若某个方法在 df 中不存在会自动跳过）
        methods_plot = [m for m, _ in csv_paths if m in set(df["method"])]

    # 画图
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=140)
    for m in methods_plot:
        sub = df[df["method"] == m].sort_values("tau")
        if sub.empty:
            continue
        x = sub["tau"].astype(float).values
        y = sub["coverage_mean"].astype(float).values
        ax.plot(x, y, marker="o", label=m)
        if show_std and "coverage_std" in sub.columns:
            sd = sub["coverage_std"].astype(float).values
            ax.fill_between(x, y - sd, y + sd, alpha=0.15)

    ax.set_xlabel("tau")
    ax.set_ylabel("coverage")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)

    # 标题
    if title is None:
        title = f"Coverage–τ | concept={concept} | guidance={guidance}"
    ax.set_title(title)
    ax.legend(loc="best", frameon=False)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[SAVE] {save_path}")
        plt.close(fig)
    else:
        plt.show()

def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", required=True, type=str, help="concept 名（如 truck）")
    ap.add_argument("--methods", required=True, type=str, help="逗号分隔方法名（如 dpp,pg,cads）")
    ap.add_argument("--guidance", required=True, type=str, help="guidance（如 5.0）")
    ap.add_argument("--outputs_root", type=str, default="/mnt/data/flow_grpo/flow_base/outputs",
                    help="根目录，默认 /mnt/data/flow_grpo/flow_base/outputs")
    ap.add_argument("--title", type=str, default=None)
    ap.add_argument("--out", type=str, default=None, help="若提供则保存到该路径；否则直接显示")
    ap.add_argument("--no_std", action="store_true", help="不画 ±std 阴影")
    ap.add_argument("--method_order", type=str, default=None, help="逗号分隔的方法顺序（可选）")
    args = ap.parse_args()

    methods = [s.strip() for s in args.methods.split(",") if s.strip()]
    method_order = [s.strip() for s in args.method_order.split(",")] if args.method_order else None

    plot_cov_tau(
        outputs_root=args.outputs_root,
        concept=args.concept,
        methods=methods,
        guidance=args.guidance,  # 保留字符串以提升文件名匹配成功率
        title=args.title,
        save_path=args.out,
        show_std=not args.no_std,
        method_order=method_order,
    )

if __name__ == "__main__":
    _main()