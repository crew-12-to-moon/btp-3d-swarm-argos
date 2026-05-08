#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def labels(xs, max_len=32):
    out = []
    for x in xs:
        s = str(x)
        out.append(s if len(s) <= max_len else s[: max_len - 3] + "...")
    return out


def bar_by(df, group, metric, out, title, ylabel, agg="mean"):
    if group not in df.columns or metric not in df.columns:
        print(f"skip {out.name}: missing {group} or {metric}")
        return
    d = df[[group, metric]].dropna()
    if d.empty:
        print(f"skip {out.name}: no data")
        return
    if agg == "mean":
        g = d.groupby(group)[metric].mean().sort_values(ascending=False)
    elif agg == "min":
        g = d.groupby(group)[metric].min().sort_values(ascending=False)
    else:
        g = d.groupby(group)[metric].max().sort_values(ascending=False)

    plt.figure(figsize=(max(10, 0.45 * len(g)), 5.5))
    plt.bar(range(len(g)), g.values)
    plt.xticks(range(len(g)), labels(g.index), rotation=60, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def box_by(df, group, metric, out, title, ylabel):
    if group not in df.columns or metric not in df.columns:
        return
    d = df[[group, metric]].dropna()
    if d.empty:
        return
    groups = sorted(d[group].unique())
    data = [d.loc[d[group] == g, metric].dropna().values for g in groups]
    groups2, data2 = [], []
    for g, arr in zip(groups, data):
        if len(arr):
            groups2.append(g)
            data2.append(arr)
    if not data2:
        return
    plt.figure(figsize=(max(10, 0.45 * len(groups2)), 5.5))
    plt.boxplot(data2, labels=labels(groups2), showmeans=True)
    plt.xticks(rotation=60, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def scatter(df, x, y, out, title, xlabel, ylabel):
    if x not in df.columns or y not in df.columns:
        return
    d = df[[x, y]].dropna()
    if d.empty:
        return
    plt.figure(figsize=(7, 5))
    plt.scatter(d[x], d[y], alpha=0.75)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def time_series(metrics_csv, out, title):
    df = pd.read_csv(metrics_csv)
    cols = [
        "step", "dist_goal", "criticality_score", "protection_infl_mean",
        "protection_decoy_mean", "lambda2_proxy", "min_neighbor_count",
        "min_decoy_decoy", "min_obstacle_decoy", "max_infl_core_dist",
        "formation_health", "mission_complete"
    ]
    df = num(df, [c for c in cols if c in df.columns])
    if "step" not in df.columns:
        return

    plots = [
        ("dist_goal", "Distance to goal", []),
        ("criticality_score", "Criticality", [0.50, 0.65]),
        ("protection_infl_mean", "Pinfl", [0.45, 0.65]),
        ("lambda2_proxy", "lambda2_proxy", [0]),
        ("min_decoy_decoy", "min_dd", [0.12, 0.20]),
        ("max_infl_core_dist", "max_infl_core_dist", [0.70, 1.10]),
        ("formation_health", "health", [0.35, 0.60]),
    ]
    plots = [(c, lab, h) for c, lab, h in plots if c in df.columns]
    if not plots:
        return

    fig, axes = plt.subplots(len(plots), 1, figsize=(10, max(9, 2.1 * len(plots))), sharex=True)
    if len(plots) == 1:
        axes = [axes]

    for ax, (c, lab, thresholds) in zip(axes, plots):
        ax.plot(df["step"], df[c])
        for th in thresholds:
            ax.axhline(th, linestyle="--", linewidth=1)
        ax.set_ylabel(lab)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Step")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def controller_comparison(df, out):
    if "controller_mode" not in df.columns:
        return
    metrics = ["success", "mean_crit", "min_Pinfl", "min_dd"]
    metrics = [m for m in metrics if m in df.columns]
    if not metrics:
        return
    g = df.groupby("controller_mode")[metrics].mean()
    plt.figure(figsize=(10, 5.5))
    x = list(range(len(g.index)))
    width = 0.8 / len(metrics)
    for i, m in enumerate(metrics):
        plt.bar([v + i * width for v in x], g[m].values, width=width, label=m)
    plt.xticks([v + width * (len(metrics) - 1) / 2 for v in x], labels(g.index), rotation=35, ha="right")
    plt.ylabel("Mean value")
    plt.title("Controller comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", help="Path to overnight_results/<timestamp>")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser().resolve()
    summary = results_dir / "summary_all.csv"
    if not summary.exists():
        raise FileNotFoundError(summary)

    plots = results_dir / "plots"
    plots.mkdir(exist_ok=True)

    df = pd.read_csv(summary)
    df = num(df, [
        "success", "time_to_goal_step", "final_dist_goal", "mean_crit", "min_crit",
        "final_crit", "mean_Pinfl", "min_Pinfl", "mean_Pdec", "mean_Jmargin",
        "min_Jmargin", "mean_lambda2_proxy", "min_lambda2_proxy",
        "disconnected_logged_rows", "mean_min_neighbors", "min_min_neighbors",
        "min_dd", "min_obstacle_decoy", "max_infl_core_dist", "mean_health",
        "min_health", "total_agents", "n_infl", "n_decoys"
    ])

    bar_by(df, "scenario", "success", plots / "summary_success_rate_by_scenario.png", "Success rate by scenario", "Success rate")
    bar_by(df, "scenario", "mean_crit", plots / "summary_mean_criticality_by_scenario.png", "Mean criticality by scenario", "Mean criticality")
    bar_by(df, "scenario", "min_Pinfl", plots / "summary_min_Pinfl_by_scenario.png", "Minimum Pinfl by scenario", "Minimum Pinfl")
    bar_by(df, "scenario", "min_lambda2_proxy", plots / "summary_min_lambda2_by_scenario.png", "Minimum connectivity proxy by scenario", "Minimum lambda2_proxy")
    bar_by(df, "scenario", "min_dd", plots / "summary_min_decoy_distance_by_scenario.png", "Minimum decoy distance by scenario", "Minimum min_dd")
    bar_by(df, "scenario", "max_infl_core_dist", plots / "summary_max_infl_core_dist_by_scenario.png", "Maximum influential-core distance by scenario", "Max influential-core distance")

    box_by(df, "scenario", "mean_crit", plots / "box_mean_criticality_by_scenario.png", "Mean criticality distribution by scenario", "Mean criticality")
    box_by(df, "scenario", "min_Pinfl", plots / "box_min_Pinfl_by_scenario.png", "Minimum Pinfl distribution by scenario", "Minimum Pinfl")

    scatter(df, "min_dd", "mean_crit", plots / "scatter_min_dd_vs_criticality.png", "Decoy spacing vs criticality", "Minimum decoy-decoy distance", "Mean criticality")
    scatter(df, "max_infl_core_dist", "mean_crit", plots / "scatter_core_spread_vs_criticality.png", "Core spread vs criticality", "Max influential-core distance", "Mean criticality")
    scatter(df, "mean_lambda2_proxy", "mean_crit", plots / "scatter_connectivity_vs_criticality.png", "Connectivity vs criticality", "Mean lambda2_proxy", "Mean criticality")

    controller_comparison(df, plots / "summary_controller_comparison.png")

    if "controller_mode" in df.columns:
        bar_by(df, "controller_mode", "success", plots / "controller_success_rate.png", "Success rate by controller", "Success rate")
        bar_by(df, "controller_mode", "mean_crit", plots / "controller_mean_criticality.png", "Mean criticality by controller", "Mean criticality")
        bar_by(df, "controller_mode", "min_Pinfl", plots / "controller_min_Pinfl.png", "Minimum Pinfl by controller", "Minimum Pinfl")

    if "run_name" in df.columns and "mean_crit" in df.columns:
        d = df.dropna(subset=["mean_crit"])
        if not d.empty:
            for tag, row in [("best", d.sort_values("mean_crit", ascending=False).iloc[0]),
                             ("worst", d.sort_values("mean_crit", ascending=True).iloc[0])]:
                run_name = row["run_name"]
                p = results_dir / str(run_name) / "btp_argos_metrics.csv"
                if p.exists():
                    time_series(p, plots / f"time_series_{tag}_{run_name}.png", f"{tag.capitalize()} run: {run_name}")

    print(f"Done. Plots saved to: {plots}")
    for p in sorted(plots.glob("*.png")):
        print(" ", p.name)


if __name__ == "__main__":
    main()
