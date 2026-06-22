"""
error_analysis_math_updated.py
Error analysis for MATH-Hard (updated run) comparison results.

Output directory: results/error_analysis/math_hard_updated/
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────

RESULTS_DIR = "results/comparison/math_updated_50"
OUTPUT_DIR  = "results/error_analysis/math_updated_50"

SYSTEMS = {
    "vanilla":   "Vanilla LLM",
    "reflexion": "Reflexion",
    "rag":       "RAG-over-Examples",
    "static":    "Static Schema",
    "passive":   "Passive Schema Memory",
    "ism":       "ISM (Ours)",
}

BASELINES = ["vanilla", "reflexion", "rag", "static", "passive"]

COLORS = {
    "vanilla":   "#d62728",
    "reflexion": "#ff7f0e",
    "rag":       "#2ca02c",
    "static":    "#9467bd",
    "passive":   "#8c564b",
    "ism":       "#1f77b4",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_logs():
    logs = {}
    for key in SYSTEMS:
        path = os.path.join(RESULTS_DIR, f"{key}.json")
        with open(path) as f:
            logs[key] = json.load(f)
    return logs


def load_ism_analysis():
    path = os.path.join(RESULTS_DIR, "ism_analysis.json")
    with open(path) as f:
        return json.load(f)


# ── Error Pattern Analysis ────────────────────────────────────────────────────

def error_patterns_by_domain(logs):
    results = {}
    for key, episodes in logs.items():
        domain_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        for ep in episodes:
            domain = ep["task"]
            domain_stats[domain]["total"] += 1
            if ep["correct"]:
                domain_stats[domain]["correct"] += 1
        results[key] = {
            d: {**v, "acc": v["correct"] / v["total"] if v["total"] > 0 else 0.0}
            for d, v in domain_stats.items()
        }
    return results


def rolling_accuracy(logs, window=5):
    """Rolling accuracy with configurable window."""
    results = {}
    for key, episodes in logs.items():
        corrects = [int(ep["correct"]) for ep in episodes]
        results[key] = [
            np.mean(corrects[max(0, i - window + 1):i + 1])
            for i in range(len(corrects))
        ]
    return results


def cumulative_accuracy(logs):
    """
    Two signals per system:
      1. cumulative_acc[i] = total correct so far / (i+1)
      2. episode_contribution[i] = correct[i] / (i+1)  — how much ep i shifted the overall acc
    """
    results = {}
    for key, episodes in logs.items():
        corrects   = [int(ep["correct"]) for ep in episodes]
        cumsum     = np.cumsum(corrects)
        n          = np.arange(1, len(corrects) + 1)
        cum_acc    = cumsum / n
        contrib    = np.array(corrects) / n   # how much each episode moved the needle
        results[key] = {"cumulative": cum_acc.tolist(),
                        "contribution": contrib.tolist()}
    return results


def domain_transition_analysis(logs):
    results = {}
    for key, episodes in logs.items():
        blocks = []
        current_domain = None
        current_block  = []
        for ep in episodes:
            if ep["task"] != current_domain:
                if current_block:
                    blocks.append((current_domain, current_block))
                current_domain = ep["task"]
                current_block  = [ep]
            else:
                current_block.append(ep)
        if current_block:
            blocks.append((current_domain, current_block))

        block_stats = []
        for domain, block in blocks:
            n         = len(block)
            early     = block[:min(10, n)]
            late      = block[max(0, n - 10):]
            early_acc = np.mean([int(e["correct"]) for e in early])
            late_acc  = np.mean([int(e["correct"]) for e in late])
            block_stats.append({
                "domain":    domain,
                "n":         n,
                "early_acc": early_acc,
                "late_acc":  late_acc,
                "delta":     late_acc - early_acc,
            })
        results[key] = block_stats
    return results


def schema_usage_analysis(logs):
    """For ISM: how often each schema was used and its accuracy."""
    ism_eps = logs.get("ism", [])
    schema_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for ep in ism_eps:
        schema = ep.get("schema", "Unknown")
        schema_stats[schema]["total"] += 1
        if ep["correct"]:
            schema_stats[schema]["correct"] += 1
    return {
        s: {**v, "acc": v["correct"] / v["total"] if v["total"] > 0 else 0.0}
        for s, v in schema_stats.items()
    }


# ── ISM Advantage Analysis ────────────────────────────────────────────────────

def compute_ism_advantage(logs):
    ism_eps  = {ep["episode"]: ep for ep in logs["ism"]}
    advantage = {}

    for key in BASELINES:
        baseline_eps  = {ep["episode"]: ep for ep in logs[key]}
        counts        = {"D": 0, "C": 0, "A": 0, "B": 0}
        domain_counts = defaultdict(lambda: {"D": 0, "C": 0, "A": 0, "B": 0})

        for ep_id, ism_ep in ism_eps.items():
            if ep_id not in baseline_eps:
                continue
            b_ep   = baseline_eps[ep_id]
            ism_ok = ism_ep["correct"]
            b_ok   = b_ep["correct"]
            domain = ism_ep["task"]

            if ism_ok and not b_ok:
                t = "D"
            elif not ism_ok and b_ok:
                t = "C"
            elif not ism_ok and not b_ok:
                t = "A"
            else:
                t = "B"

            counts[t] += 1
            domain_counts[domain][t] += 1

        advantage[key] = {
            "overall":        counts,
            "by_domain":      dict(domain_counts),
            "net_advantage":  counts["D"] - counts["C"],
            "advantage_rate": counts["D"] / max(1, counts["D"] + counts["C"]),
        }
    return advantage


def summarize_advantage(advantage):
    summary = []
    for key in BASELINES:
        a = advantage[key]
        summary.append({
            "baseline":       SYSTEMS[key],
            "ISM_wins":       a["overall"]["D"],
            "ISM_loses":      a["overall"]["C"],
            "both_wrong":     a["overall"]["A"],
            "both_correct":   a["overall"]["B"],
            "net_advantage":  a["net_advantage"],
            "advantage_rate": round(a["advantage_rate"], 3),
            "by_domain": {
                d: {"ISM_wins": v["D"], "ISM_loses": v["C"]}
                for d, v in a["by_domain"].items()
            },
        })
    return summary


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_rolling_accuracy(rolling_results, logs, output_dir, window=1):
    fig, ax = plt.subplots(figsize=(16, 5))

    ism_tasks  = [ep["task"] for ep in logs["ism"]]
    boundaries = []
    prev_task  = ism_tasks[0]
    for i, t in enumerate(ism_tasks):
        if t != prev_task:
            boundaries.append(i)
            prev_task = t

    for b in boundaries:
        ax.axvline(b, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    # Label domain regions
    prev_b    = 0
    prev_task = ism_tasks[0]
    for b in boundaries + [len(ism_tasks)]:
        mid = (prev_b + b) / 2
        ax.text(mid, 1.03, prev_task, ha="center", fontsize=8,
                transform=ax.get_xaxis_transform())
        prev_b    = b
        prev_task = ism_tasks[b] if b < len(ism_tasks) else ""

    for key, values in rolling_results.items():
        lw = 2.5 if key == "ism" else 1.2
        alpha = 1.0 if key == "ism" else 0.75
        ax.plot(values, label=SYSTEMS[key], color=COLORS[key],
                linewidth=lw, alpha=alpha)

    ylabel = "Accuracy (per episode)" if window == 1 else f"Rolling Accuracy (window={window})"
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    title = ("MATH-Hard: Per-Episode Accuracy" if window == 1
             else f"MATH-Hard: Rolling Accuracy (window={window})")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlim(0, len(ism_tasks) - 1)
    ax.legend(fontsize=9, loc="lower right", ncol=2)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    fname = "per_episode_accuracy.png" if window == 1 else f"rolling_accuracy_w{window}.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


def plot_accuracy_by_domain(domain_results, output_dir):
    domains = sorted({d for v in domain_results.values() for d in v})
    systems = list(SYSTEMS.keys())
    x       = np.arange(len(domains))
    width   = 0.13

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, key in enumerate(systems):
        accs = [domain_results[key].get(d, {}).get("acc", 0.0) for d in domains]
        bars = ax.bar(x + i * width, accs, width, label=SYSTEMS[key],
                      color=COLORS[key], alpha=0.85)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{acc:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Domain", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("MATH-Hard: Accuracy by Domain per System", fontsize=13, fontweight="bold")
    ax.set_xticks(x + width * (len(systems) - 1) / 2)
    ax.set_xticklabels(domains, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_by_domain.png"), dpi=150)
    plt.close()
    print("  Saved: accuracy_by_domain.png")


def plot_ism_advantage_bars(advantage, output_dir):
    labels = [SYSTEMS[k] for k in BASELINES]
    wins   = [advantage[k]["overall"]["D"] for k in BASELINES]
    loses  = [advantage[k]["overall"]["C"] for k in BASELINES]
    net    = [advantage[k]["net_advantage"] for k in BASELINES]
    x      = np.arange(len(BASELINES))
    width  = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.bar(x - width/2, wins,  width, label="ISM Wins",  color="#1f77b4", alpha=0.85)
    ax.bar(x + width/2, loses, width, label="ISM Loses", color="#d62728", alpha=0.85)
    for xi, w, l in zip(x, wins, loses):
        ax.text(xi - width/2, w + 0.3, str(w), ha="center", fontsize=9, fontweight="bold")
        ax.text(xi + width/2, l + 0.3, str(l), ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Episode Count", fontsize=11)
    ax.set_title("ISM Wins vs Losses per Baseline", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    ax2 = axes[1]
    bar_colors = ["#1f77b4" if n >= 0 else "#d62728" for n in net]
    bars = ax2.bar(x, net, color=bar_colors, alpha=0.85)
    for xi, n in zip(x, net):
        offset = 0.3 if n >= 0 else -1.2
        ax2.text(xi, n + offset, f"{n:+d}", ha="center", fontsize=10, fontweight="bold")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax2.set_ylabel("Net Advantage (Wins − Losses)", fontsize=11)
    ax2.set_title("ISM Net Advantage over Each Baseline", fontsize=11, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    plt.suptitle("MATH-Hard (Updated): ISM Advantage Analysis",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ism_advantage.png"), dpi=150)
    plt.close()
    print("  Saved: ism_advantage.png")


def plot_ism_advantage_by_domain(advantage, output_dir):
    all_domains = sorted({
        d for key in BASELINES for d in advantage[key]["by_domain"]
    })
    n_domains = len(all_domains)
    fig, axes = plt.subplots(1, n_domains, figsize=(5 * n_domains, 5), sharey=False)
    if n_domains == 1:
        axes = [axes]

    for ax, domain in zip(axes, all_domains):
        wins  = [advantage[k]["by_domain"].get(domain, {}).get("ISM_wins",  0) for k in BASELINES]
        loses = [advantage[k]["by_domain"].get(domain, {}).get("ISM_loses", 0) for k in BASELINES]
        x     = np.arange(len(BASELINES))
        width = 0.35

        ax.bar(x - width/2, wins,  width, label="ISM Wins",  color="#1f77b4", alpha=0.85)
        ax.bar(x + width/2, loses, width, label="ISM Loses", color="#d62728", alpha=0.85)
        ax.set_title(domain, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([SYSTEMS[k].replace(" ", "\n") for k in BASELINES], fontsize=7)
        ax.set_ylabel("Episodes")
        ax.legend(fontsize=7)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.suptitle("MATH-Hard: ISM Advantage by Domain vs Each Baseline",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ism_advantage_by_domain.png"), dpi=150)
    plt.close()
    print("  Saved: ism_advantage_by_domain.png")


def plot_error_heatmap(domain_results, output_dir):
    domains = sorted({d for v in domain_results.values() for d in v})
    systems = list(SYSTEMS.keys())

    matrix = np.array([
        [1.0 - domain_results[k].get(d, {}).get("acc", 0.0) for d in domains]
        for k in systems
    ])

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, cmap="Reds", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels(domains, fontsize=10)
    ax.set_yticks(range(len(systems)))
    ax.set_yticklabels([SYSTEMS[k] for k in systems], fontsize=9)
    for i in range(len(systems)):
        for j in range(len(domains)):
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                    fontsize=9, color="white" if matrix[i,j] > 0.5 else "black")
    plt.colorbar(im, ax=ax, label="Error Rate")
    ax.set_title("MATH-Hard: Error Rate Heatmap (System × Domain)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_heatmap.png"), dpi=150)
    plt.close()
    print("  Saved: error_heatmap.png")


def plot_plasticity_stability(transition_results, output_dir):
    systems = list(SYSTEMS.keys())
    all_domains = sorted({
        b["domain"] for key in systems for b in transition_results[key]
    })
    x     = np.arange(len(all_domains))
    width = 0.13

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax_idx, (metric, label) in enumerate([
        ("early_acc", "Plasticity (first 10 eps)"),
        ("late_acc",  "Stability (last 10 eps)"),
    ]):
        ax = axes[ax_idx]
        for i, key in enumerate(systems):
            domain_vals = defaultdict(list)
            for b in transition_results[key]:
                domain_vals[b["domain"]].append(b[metric])
            vals = [np.mean(domain_vals.get(d, [0.0])) for d in all_domains]
            ax.bar(x + i * width, vals, width, label=SYSTEMS[key],
                   color=COLORS[key], alpha=0.85)
        ax.set_xticks(x + width * (len(systems) - 1) / 2)
        ax.set_xticklabels(all_domains, fontsize=10)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(f"MATH-Hard: {label}", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=7)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plasticity_stability.png"), dpi=150)
    plt.close()
    print("  Saved: plasticity_stability.png")


def plot_schema_usage(schema_stats, ism_analysis, output_dir):
    """ISM-specific: schema usage count and accuracy, with seed vs new labeling."""
    seed_names = {
        "Algebra", "Number Theory", "Geometry",
        "Combinatorics", "Probability", "Calculus and Analysis"
    }

    # Sort by usage count
    items = sorted(schema_stats.items(), key=lambda x: -x[1]["total"])
    names  = [s for s, _ in items]
    usages = [v["total"] for _, v in items]
    accs   = [v["acc"]   for _, v in items]
    colors = ["#1f77b4" if n in seed_names else "#ff7f0e" for n in names]

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    # Top: usage count
    ax = axes[0]
    bars = ax.bar(range(len(names)), usages, color=colors, alpha=0.85)
    for bar, val in zip(bars, usages):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(val), ha="center", fontsize=8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Times Used", fontsize=11)
    ax.set_title("ISM Schema Usage Count (blue=seed, orange=synthesized)",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    seed_patch = mpatches.Patch(color="#1f77b4", label="Seed schema")
    new_patch  = mpatches.Patch(color="#ff7f0e", label="Synthesized schema")
    ax.legend(handles=[seed_patch, new_patch], fontsize=9)

    # Bottom: accuracy per schema
    ax2 = axes[1]
    acc_colors = ["#2ca02c" if a >= 0.7 else "#ff7f0e" if a >= 0.4 else "#d62728"
                  for a in accs]
    bars2 = ax2.bar(range(len(names)), accs, color=acc_colors, alpha=0.85)
    for bar, acc in zip(bars2, accs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{acc:.2f}", ha="center", fontsize=8)
    ax2.axhline(0.7, color="green",  linestyle="--", alpha=0.5, label="Strong (0.7)")
    ax2.axhline(0.4, color="orange", linestyle="--", alpha=0.5, label="Weak (0.4)")
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.set_ylim(0, 1.15)
    ax2.set_title("ISM Schema Accuracy (green≥0.7, orange=0.4–0.7, red<0.4)",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ism_schema_usage.png"), dpi=150)
    plt.close()
    print("  Saved: ism_schema_usage.png")


def plot_improvement_timeline(ism_analysis, output_dir):
    """Plot ISM self-improvement events over time."""
    improvement_log = ism_analysis.get("improvement_log", [])
    if not improvement_log:
        print("  Skipped: improvement_timeline.png (no improvement log)")
        return

    mechanism_colors = {
        "Self-Audit":          "#aec7e8",
        "Self-Correct":        "#ffbb78",
        "Self-Merge":          "#98df8a",
        "Self-Prune":          "#ff9896",
        "Self-Escalate":       "#c5b0d5",
        "Self-Promote/Demote": "#c49c94",
    }

    mechanisms = [e["mechanism"] for e in improvement_log]
    episodes   = [e["episode"]   for e in improvement_log]

    unique_mechs = list(mechanism_colors.keys())
    y_pos = {m: i for i, m in enumerate(unique_mechs)}

    fig, ax = plt.subplots(figsize=(16, 5))
    for ep, mech in zip(episodes, mechanisms):
        color = mechanism_colors.get(mech, "gray")
        y     = y_pos.get(mech, len(unique_mechs))
        ax.scatter(ep, y, color=color, s=80, zorder=3, alpha=0.9)

    ax.set_yticks(range(len(unique_mechs)))
    ax.set_yticklabels(unique_mechs, fontsize=10)
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_title("ISM Self-Improvement Events over Episode Stream",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(-2, 182)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    # Domain boundary lines
    ax.axvline(30,  color="gray", linestyle="--", alpha=0.4)
    ax.axvline(60,  color="gray", linestyle="--", alpha=0.4)
    ax.axvline(90,  color="gray", linestyle="--", alpha=0.4)
    ax.axvline(120, color="gray", linestyle="--", alpha=0.4)
    ax.axvline(150, color="gray", linestyle="--", alpha=0.4)

    domains = ["Algebra", "Num.Th.", "Geometry", "Num.Th.", "Algebra"]
    for i, (start, end, name) in enumerate(zip(
        [0, 30, 60, 90, 120, 150],
        [30, 60, 90, 120, 150, 180],
        ["Algebra", "Num.Th.", "Geometry", "Num.Th.(hard)", "Algebra", "Algebra"],
    )):
        ax.text((start + end) / 2, len(unique_mechs) - 0.3, name,
                ha="center", fontsize=7, color="gray")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "improvement_timeline.png"), dpi=150)
    plt.close()
    print("  Saved: improvement_timeline.png")


def plot_bank_size_evolution(logs, output_dir):
    """ISM vs Passive bank size over episodes."""
    fig, ax = plt.subplots(figsize=(14, 4))

    for key in ["ism", "passive", "static"]:
        if key not in logs:
            continue
        sizes = [ep["bank_size"] for ep in logs[key]]
        lw    = 2.5 if key == "ism" else 1.5
        ax.plot(sizes, label=SYSTEMS[key], color=COLORS[key], linewidth=lw)

    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Schema Bank Size", fontsize=11)
    ax.set_title("Bank Size Evolution: ISM vs Passive vs Static",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(linestyle="--", alpha=0.3)

    for b in [30, 60, 90, 120, 150]:
        ax.axvline(b, color="gray", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bank_size_evolution.png"), dpi=150)
    plt.close()
    print("  Saved: bank_size_evolution.png")


def plot_cumulative_accuracy(cum_results, logs, output_dir):
    """
    Top panel:    cumulative accuracy per system (total correct / total episodes so far)
    Bottom panel: per-episode contribution to overall accuracy (correct[i] / (i+1))
                  shown as bar chart — green=correct, red=incorrect
    """
    ism_tasks  = [ep["task"] for ep in logs["ism"]]
    boundaries = []
    prev_task  = ism_tasks[0]
    for i, t in enumerate(ism_tasks):
        if t != prev_task:
            boundaries.append(i)
            prev_task = t

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

    # ── Top: cumulative accuracy ──────────────────────────────────────────
    ax = axes[0]
    for b in boundaries:
        ax.axvline(b, color="gray", linestyle="--", alpha=0.4)

    prev_b    = 0
    prev_task = ism_tasks[0]
    for b in boundaries + [len(ism_tasks)]:
        mid = (prev_b + b) / 2
        ax.text(mid, 1.04, prev_task, ha="center", fontsize=8,
                transform=ax.get_xaxis_transform())
        prev_b    = b
        prev_task = ism_tasks[b] if b < len(ism_tasks) else ""

    for key, data in cum_results.items():
        lw    = 2.5 if key == "ism" else 1.2
        alpha = 1.0 if key == "ism" else 0.75
        ax.plot(data["cumulative"], label=SYSTEMS[key],
                color=COLORS[key], linewidth=lw, alpha=alpha)

    # Mark final accuracy values on the right
    for key, data in cum_results.items():
        final = data["cumulative"][-1]
        ax.annotate(f"{final:.2f}",
                    xy=(len(data["cumulative"]) - 1, final),
                    xytext=(5, 0), textcoords="offset points",
                    fontsize=7, color=COLORS[key], va="center")

    ax.set_ylabel("Cumulative Accuracy\n(total correct / total episodes)", fontsize=10)
    ax.set_title("MATH-Hard: Cumulative Accuracy & Per-Episode Contribution",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=9, loc="lower right", ncol=2)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # ── Bottom: ISM per-episode contribution ─────────────────────────────
    ax2 = axes[1]
    for b in boundaries:
        ax2.axvline(b, color="gray", linestyle="--", alpha=0.4)

    ism_eps      = logs["ism"]
    ism_corrects = [int(ep["correct"]) for ep in ism_eps]
    n            = np.arange(1, len(ism_corrects) + 1)
    contrib      = np.array(ism_corrects) / n

    bar_colors = ["#2ca02c" if c else "#d62728" for c in ism_corrects]
    ax2.bar(range(len(contrib)), contrib, color=bar_colors, alpha=0.7, width=1.0)

    ax2.set_xlabel("Episode", fontsize=11)
    ax2.set_ylabel("Episode Contribution\n(correct[i] / (i+1))", fontsize=10)
    ax2.set_title("ISM: How Each Episode Contributes to Overall Accuracy\n"
                  "(green = correct, red = incorrect)", fontsize=11)
    ax2.set_ylim(0, None)
    ax2.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cumulative_accuracy.png"), dpi=150)
    plt.close()
    print("  Saved: cumulative_accuracy.png")


def plot_schema_health_over_time(ism_analysis, output_dir):
    """Stacked area of schema health counts per audit."""
    health_log = ism_analysis.get("health_log", [])
    if not health_log:
        print("  Skipped: schema_health_over_time.png (no health log)")
        return

    episodes = [h["episode"] for h in health_log]
    strong   = [h["report"] and sum(1 for v in h["report"].values() if v == "strong")  for h in health_log]
    neutral  = [h["report"] and sum(1 for v in h["report"].values() if v == "neutral") for h in health_log]
    weak     = [h["report"] and sum(1 for v in h["report"].values() if v == "weak")    for h in health_log]
    unused   = [h["report"] and sum(1 for v in h["report"].values() if v == "unused")  for h in health_log]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.stackplot(episodes, strong, neutral, weak, unused,
                 labels=["Strong", "Neutral", "Weak", "Unused"],
                 colors=["#2ca02c", "#aec7e8", "#ff7f0e", "#d62728"],
                 alpha=0.8)
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Schema Count", fontsize=11)
    ax.set_title("ISM Schema Health Distribution over Time",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "schema_health_over_time.png"), dpi=150)
    plt.close()
    print("  Saved: schema_health_over_time.png")


# ── Save JSON Report ──────────────────────────────────────────────────────────

def save_report(domain_results, advantage_summary, transition_results,
                schema_stats, output_dir):
    report = {
        "dataset": "MATH-Hard (Level 4-5) — Updated Run",
        "accuracy_by_domain": {
            key: {
                domain: {
                    "accuracy": round(v["acc"], 4),
                    "correct":  v["correct"],
                    "total":    v["total"],
                }
                for domain, v in domain_results[key].items()
            }
            for key in SYSTEMS
        },
        "ism_advantage_over_baselines": advantage_summary,
        "block_transition_analysis": {
            key: [
                {k: (round(v, 4) if isinstance(v, float) else v) for k, v in b.items()}
                for b in blocks
            ]
            for key, blocks in transition_results.items()
        },
        "ism_schema_stats": {
            s: {"accuracy": round(v["acc"], 4), "uses": v["total"], "correct": v["correct"]}
            for s, v in sorted(schema_stats.items(), key=lambda x: -x[1]["total"])
        },
    }
    path = os.path.join(output_dir, "error_analysis_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved: error_analysis_report.json")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Error Analysis — MATH-Hard (Updated Run)")
    print("=" * 60)

    logs         = load_logs()
    ism_analysis = load_ism_analysis()
    print(f"Loaded {len(logs)} systems, {len(logs['ism'])} episodes each\n")

    print("[1] Error patterns by domain...")
    domain_results = error_patterns_by_domain(logs)
    for key in SYSTEMS:
        print(f"  {SYSTEMS[key]:<30}", end="")
        for d, v in sorted(domain_results[key].items()):
            print(f"  {d}: {v['acc']:.3f}", end="")
        print()

    print("\n[2] Rolling accuracy (window=5) + cumulative accuracy...")
    rolling_results = rolling_accuracy(logs, window=5)
    cum_results     = cumulative_accuracy(logs)

    print("\n[3] Block transition analysis...")
    transition_results = domain_transition_analysis(logs)

    print("\n[4] ISM advantage over baselines...")
    advantage   = compute_ism_advantage(logs)
    adv_summary = summarize_advantage(advantage)
    for row in adv_summary:
        print(f"  vs {row['baseline']:<30} "
              f"Wins: {row['ISM_wins']:>3}  "
              f"Loses: {row['ISM_loses']:>3}  "
              f"Net: {row['net_advantage']:>+4}  "
              f"Rate: {row['advantage_rate']:.3f}")

    print("\n[5] ISM schema usage analysis...")
    schema_stats = schema_usage_analysis(logs)
    for s, v in sorted(schema_stats.items(), key=lambda x: -x[1]["total"]):
        print(f"  {s:<40} uses={v['total']:>3}  acc={v['acc']:.3f}")

    print("\n[6] Generating plots...")
    plot_rolling_accuracy(rolling_results, logs, OUTPUT_DIR, window=5)
    plot_cumulative_accuracy(cum_results, logs, OUTPUT_DIR)
    plot_accuracy_by_domain(domain_results, OUTPUT_DIR)
    plot_ism_advantage_bars(advantage, OUTPUT_DIR)
    plot_ism_advantage_by_domain(advantage, OUTPUT_DIR)
    plot_error_heatmap(domain_results, OUTPUT_DIR)
    plot_plasticity_stability(transition_results, OUTPUT_DIR)
    plot_schema_usage(schema_stats, ism_analysis, OUTPUT_DIR)
    plot_improvement_timeline(ism_analysis, OUTPUT_DIR)
    plot_bank_size_evolution(logs, OUTPUT_DIR)
    plot_schema_health_over_time(ism_analysis, OUTPUT_DIR)

    print("\n[7] Saving JSON report...")
    save_report(domain_results, adv_summary, transition_results,
                schema_stats, OUTPUT_DIR)

    print(f"\nDone. All outputs saved to: {OUTPUT_DIR}")
    print(f"\nPlots generated:")
    print(f"  rolling_accuracy_w5.png     — rolling accuracy window=5")
    print(f"  cumulative_accuracy.png     — cumulative acc + per-episode contribution")
    print(f"  accuracy_by_domain.png      — accuracy per domain per system")
    print(f"  ism_advantage.png           — ISM wins/losses vs each baseline")
    print(f"  ism_advantage_by_domain.png — ISM advantage broken down by domain")
    print(f"  error_heatmap.png           — error rate heatmap (system × domain)")
    print(f"  plasticity_stability.png    — first vs last 10 eps per domain")
    print(f"  ism_schema_usage.png        — schema usage count + accuracy")
    print(f"  improvement_timeline.png    — self-improvement events over time")
    print(f"  bank_size_evolution.png     — ISM vs Passive bank size over time")
    print(f"  schema_health_over_time.png — schema health distribution per audit")


if __name__ == "__main__":
    main()
