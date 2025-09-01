# plot_cov_tau.py
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def plot_cov_tau(
    csv_paths,
    concept: str = None,
    guidance: float = None,
    title: str = None,
    save_path: str = None,
    show_std: bool = True,
    method_order=None,
):
    """
    读取一个或多个 'exp3_{guidance}_{concept}.csv' 并绘制 Coverage–τ 曲线。
    期望列：method, concept, guidance, k, tau, coverage_mean, coverage_std
    """
    # 1) 读入并合并
    if isinstance(csv_paths, (str, Path)):
        csv_paths = [csv_paths]
    frames = []
    for p in csv_paths:
        df = pd.read_csv(p)
        df["__src__"] = str(p)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    # 2) 过滤 concept / guidance（如果指定）
    if concept is not None:
        df = df[df["concept"].astype(str).str.lower() == str(concept).lower()]
    if guidance is not None:
        df = df[df["guidance"].astype(float) == float(guidance)]

    if df.empty:
        raise ValueError("过滤后没有数据（检查 concept / guidance / CSV 内容）")

    # 3) 基本检查：guidance 是否一致
    guids = sorted(df["guidance"].unique())
    if len(guids) > 1:
        print(f"[WARN] 读入了多个 guidance: {guids}，将全部画在同一图上（如果只想要单一 guidance，请设置 guidance=...）")

    # 4) 排序并去重（防止重复读入同一方法同一行）
    df = df.sort_values(["method", "tau"]).drop_duplicates(["method","tau","guidance","concept"])

    # 5) 画图
    fig, ax = plt.subplots(figsize=(6,4.2), dpi=140)

    # 方法顺序
    methods = list(df["method"].unique()) if method_order is None else [m for m in method_order if m in set(df["method"])]

    for m in methods:
        sub = df[df["method"] == m].sort_values("tau")
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
        tt = []
        if concept: tt.append(f"concept={concept}")
        if guidance is not None: tt.append(f"guidance={guidance}")
        title = "Coverage–τ " + (" | ".join(tt) if tt else "")
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
    ap.add_argument("--csv", nargs="+", required=True, help="一个或多个 exp3_{guidance}_{concept}.csv 路径")
    ap.add_argument("--concept", type=str, default=None, help="可选：只画某个 concept")
    ap.add_argument("--guidance", type=float, default=None, help="可选：只画某个 guidance（如 5.0）")
    ap.add_argument("--title", type=str, default=None)
    ap.add_argument("--out", type=str, default=None, help="若提供则保存到该路径；否则直接显示")
    ap.add_argument("--no_std", action="store_true", help="不画 ±std 阴影")
    ap.add_argument("--method_order", type=str, default=None, help="逗号分隔的方法顺序")
    args = ap.parse_args()

    method_order = [s.strip() for s in args.method_order.split(",")] if args.method_order else None

    plot_cov_tau(
        csv_paths=args.csv,
        concept=args.concept,
        guidance=args.guidance,
        title=args.title,
        save_path=args.out,
        show_std=not args.no_std,
        method_order=method_order,
    )

if __name__ == "__main__":
    _main()
