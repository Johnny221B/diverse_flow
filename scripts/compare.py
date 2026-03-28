import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_metrics_per_prompt(
    methods,
    category,
    base_dir="/data2/toby/OSCAR/outputs",
    metrics=None,
    x_key="prompt_text",
    save_path=None,
    avg_save_path=None,
    sort_x=False,
    figsize=(24, 16),
):
    """
    根据 methods 和 category 读取多个 csv，并画每个 prompt 的多方法对比图。
    同时计算每个方法在各个 metric 上的平均值。

    路径格式:
        /data2/toby/OSCAR/outputs/{method}_{category}/eval/metrics_per_prompt.csv
    """
    if metrics is None:
        metrics = ["vendi_inception", "vendi_pixel", "clip_score", "one_minus_ms_ssim"]

    method_dfs = {}
    all_x = []
    avg_rows = []

    # 读取所有方法的数据
    for method in methods:
        csv_path = os.path.join(
            base_dir,
            f"{method}_{category}",
            "eval",
            "metrics_per_prompt.csv"
        )

        if not os.path.exists(csv_path):
            print(f"[Warning] 文件不存在，跳过: {csv_path}")
            continue

        df = pd.read_csv(csv_path)

        required_cols = [x_key] + metrics
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            print(f"[Warning] {method} 缺少列 {missing_cols}，跳过")
            continue

        df = df[[x_key] + metrics].copy()
        df = df.drop_duplicates(subset=[x_key])

        if sort_x:
            df = df.sort_values(by=x_key)

        method_dfs[method] = df
        all_x.extend(df[x_key].tolist())

        # 计算平均值
        avg_dict = {"method": method}
        for metric in metrics:
            avg_dict[metric] = df[metric].mean()
        avg_rows.append(avg_dict)

    if len(method_dfs) == 0:
        raise ValueError("没有成功读取任何方法的数据，请检查 methods 和 category。")

    # 输出平均值表
    avg_df = pd.DataFrame(avg_rows)
    print("\n=== Average Metrics ===")
    print(avg_df.to_string(index=False))

    # 如果需要，保存平均值 csv
    if avg_save_path is not None:
        os.makedirs(os.path.dirname(avg_save_path), exist_ok=True)
        avg_df.to_csv(avg_save_path, index=False)
        print(f"\n平均值已保存到: {avg_save_path}")

    # 统一横轴顺序
    all_x = list(dict.fromkeys(all_x))
    if sort_x:
        all_x = sorted(all_x)

    # 创建 2x2 子图
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        for method, df in method_dfs.items():
            value_map = dict(zip(df[x_key], df[metric]))
            y = [value_map.get(x, float("nan")) for x in all_x]
            ax.plot(range(len(all_x)), y, marker="o", linewidth=2, label=method)

        ax.set_title(metric, fontsize=14)
        ax.set_xlabel(x_key, fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax.set_xticks(range(len(all_x)))
        ax.set_xticklabels(all_x, rotation=75, ha="right", fontsize=8)

    plt.suptitle(f"Per-prompt Metrics Comparison ({category})", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\n图已保存到: {save_path}")

    plt.show()

    return avg_df


if __name__ == "__main__":
    methods = ["base", "ourmethod", "cads", "pg", "dpp", "mix", "apg"]
    category = "t2i_small"

    avg_df = plot_metrics_per_prompt(
        methods=methods,
        category=category,
        save_path=f"/data2/toby/OSCAR/outputs/results/{category}.png",
        avg_save_path=f"/data2/toby/OSCAR/outputs/results/{category}_average.csv",
        sort_x=False,
    )

