"""
error_analysis_olympiad.py
Error analysis for OlympiadBench comparison results.

Loads per-episode logs for all 6 systems, identifies error patterns,
computes ISM advantage over each baseline, and generates plots.

Output directory: results/error_analysis/olympiad/
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────

RESULTS_DIR = "results/comparison/olympiad_updated_50_new"
OUTPUT_DIR  = "results/error_analysis/olympiad_50_new"

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
    if not os.path.exists(path):
        print("  [WARN] ism_analysis.json not found — skipping ISM-specific plots")
        return None
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


def error_patterns_by_block(logs):
    results = {}
    for key, episodes in logs.items():
        corrects = [int(ep["correct"]) for ep in episodes]
        window   = 10
        rolling  = [
            np.mean(corrects[max(0, i - window):i + 1])
            for i in range(len(corrects))
        ]
        results[key] = rolling
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


def hardest_domains(domain_results):
    """
    Identify domains where ALL systems struggle (avg error rate > 0.5).
    OlympiadBench-specific — these are the hardest subfields.
    """
    all_domains = sorted({d for v in domain_results.values() for d in v})
    hard = []
    for domain in all_domains:
        avg_err = np.mean([
            1.0 - domain_results[key].get(domain, {}).get("acc", 0.0)
            for key in SYSTEMS
        ])
        hard.append((domain, avg_err))
    hard.sort(key=lambda x: -x[1])
    return hard


# ── ISM Advantage Analysis ────────────────────────────────────────────────────

def compute_ism_advantage(logs):
    ism_eps  = {ep["episode"]: ep for ep in logs["ism"]}
    advantage = {}

    for key in BASELINES:
        baseline_eps = {ep["episode"]: ep for ep in logs[key]}
        counts       = {"D": 0, "C": 0, "A": 0, "B": 0}
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
            "overall":       counts,
            "by_domain":     dict(domain_counts),
            "net_advantage": counts["D"] - counts["C"],
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
            "by_domain":      {
                d: {
                    "ISM_wins":  v["D"],
                    "ISM_loses": v["C"],
                }
                for d, v in a["by_domain"].items()
            },
        })
    return summary


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_accuracy_by_domain(domain_results, output_dir):
    domains = sorted({d for v in domain_results.values() for d in v})
    systems = list(SYSTEMS.keys())
    x       = np.arange(len(domains))
    width   = 0.13

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, key in enumerate(systems):
        accs = [domain_results[key].get(d, {}).get("acc", 0.0) for d in domains]
        bars = ax.bar(x + i * width, accs, width, label=SYSTEMS[key],
                      color=COLORS[key], alpha=0.85)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{acc:.2f}", ha="center", va="bottom", fontsize=6.5)

    ax.set_xlabel("Domain")
    ax.set_ylabel("Accuracy")
    ax.set_title("OlympiadBench: Accuracy by Domain per System")
    ax.set_xticks(x + width * (len(systems) - 1) / 2)
    ax.set_xticklabels(domains)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_by_domain.png"), dpi=150)
    plt.close()
    print("  Saved: accuracy_by_domain.png")


def plot_rolling_accuracy(rolling_results, logs, output_dir):
    fig, ax = plt.subplots(figsize=(14, 5))

    ism_tasks  = [ep["task"] for ep in logs["ism"]]
    boundaries = []
    prev_task  = ism_tasks[0]
    for i, t in enumerate(ism_tasks):
        if t != prev_task:
            boundaries.append(i)
            prev_task = t
    for b in boundaries:
        ax.axvline(b, color="gray", linestyle="--", alpha=0.4)

    prev_b    = 0
    prev_task = ism_tasks[0]
    for b in boundaries + [len(ism_tasks)]:
        mid = (prev_b + b) / 2
        ax.text(mid, 1.02, prev_task, ha="center", fontsize=7,
                transform=ax.get_xaxis_transform())
        prev_b    = b
        prev_task = ism_tasks[b] if b < len(ism_tasks) else ""

    for key, rolling in rolling_results.items():
        lw = 2.5 if key == "ism" else 1.2
        ax.plot(rolling, label=SYSTEMS[key], color=COLORS[key],
                linewidth=lw, alpha=0.9)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Rolling Accuracy (window=10)")
    ax.set_title("OlympiadBench: Rolling Accuracy over Episode Stream")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rolling_accuracy.png"), dpi=150)
    plt.close()
    print("  Saved: rolling_accuracy.png")


def plot_ism_advantage_bars(advantage, output_dir):
    baselines = BASELINES
    labels    = [SYSTEMS[k] for k in baselines]
    wins      = [advantage[k]["overall"]["D"] for k in baselines]
    loses     = [advantage[k]["overall"]["C"] for k in baselines]
    net       = [advantage[k]["net_advantage"] for k in baselines]

    x     = np.arange(len(baselines))
    width = 0.28

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.bar(x - width / 2, wins,  width, label="ISM Wins (D)",  color="#1f77b4", alpha=0.85)
    ax.bar(x + width / 2, loses, width, label="ISM Loses (C)", color="#d62728", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Episode Count")
    ax.set_title("ISM Wins vs Losses per Baseline")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    ax2 = axes[1]
    bar_colors = ["#1f77b4" if n >= 0 else "#d62728" for n in net]
    ax2.bar(x, net, color=bar_colors, alpha=0.85)
    for xi, n in zip(x, net):
        ax2.text(xi, n + (0.3 if n >= 0 else -1.0), str(n),
                 ha="center", fontsize=9, fontweight="bold")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax2.set_ylabel("Net Advantage (Wins − Losses)")
    ax2.set_title("ISM Net Advantage over Each Baseline")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    plt.suptitle("OlympiadBench: ISM Advantage Analysis", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ism_advantage.png"), dpi=150)
    plt.close()
    print("  Saved: ism_advantage.png")


def plot_ism_advantage_by_domain(advantage, output_dir):
    all_domains = sorted({
        d
        for key in BASELINES
        for d in advantage[key]["by_domain"]
    })

    n_baselines = len(BASELINES)
    n_domains   = len(all_domains)
    fig, axes   = plt.subplots(1, n_domains, figsize=(5 * n_domains, 5), sharey=False)
    if n_domains == 1:
        axes = [axes]

    for ax, domain in zip(axes, all_domains):
        wins  = [advantage[k]["by_domain"].get(domain, {}).get("ISM_wins",  0) for k in BASELINES]
        loses = [advantage[k]["by_domain"].get(domain, {}).get("ISM_loses", 0) for k in BASELINES]
        x     = np.arange(n_baselines)
        width = 0.35

        ax.bar(x - width / 2, wins,  width, label="ISM Wins",  color="#1f77b4", alpha=0.85)
        ax.bar(x + width / 2, loses, width, label="ISM Loses", color="#d62728", alpha=0.85)
        ax.set_title(domain, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([SYSTEMS[k].replace(" ", "\n") for k in BASELINES],
                           fontsize=7)
        ax.set_ylabel("Episodes")
        ax.legend(fontsize=7)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.suptitle("OlympiadBench: ISM Advantage by Domain vs Each Baseline",
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

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(matrix, cmap="Reds", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels(domains, fontsize=10)
    ax.set_yticks(range(len(systems)))
    ax.set_yticklabels([SYSTEMS[k] for k in systems], fontsize=9)

    for i in range(len(systems)):
        for j in range(len(domains)):
            ax.text(j, i, f"{matrix[i, j]:.2f}",
                    ha="center", va="center", fontsize=9,
                    color="white" if matrix[i, j] > 0.5 else "black")

    plt.colorbar(im, ax=ax, label="Error Rate")
    ax.set_title("OlympiadBench: Error Rate Heatmap (by System × Domain)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_heatmap.png"), dpi=150)
    plt.close()
    print("  Saved: error_heatmap.png")


def plot_hardest_domains(domain_results, output_dir):
    hard = hardest_domains(domain_results)
    domains = [d for d, _ in hard]
    avg_errors = [e for _, e in hard]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(domains, avg_errors, color="#d62728", alpha=0.8)
    for bar, val in zip(bars, avg_errors):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    ax.set_xlabel("Average Error Rate (across all systems)")
    ax.set_title("OlympiadBench: Hardest Domains (avg error rate across all systems)")
    ax.set_xlim(0, 1.1)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hardest_domains.png"), dpi=150)
    plt.close()
    print("  Saved: hardest_domains.png")


def plot_block_plasticity_stability(transition_results, output_dir):
    systems = list(SYSTEMS.keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, metric in enumerate(["early_acc", "late_acc"]):
        ax    = axes[ax_idx]
        label = "Plasticity (first 10 eps)" if metric == "early_acc" else "Stability (last 10 eps)"

        all_domains = sorted({
            b["domain"]
            for key in systems
            for b in transition_results[key]
        })

        x     = np.arange(len(all_domains))
        width = 0.13

        for i, key in enumerate(systems):
            domain_vals = defaultdict(list)
            for b in transition_results[key]:
                domain_vals[b["domain"]].append(b[metric])
            vals = [np.mean(domain_vals.get(d, [0.0])) for d in all_domains]
            ax.bar(x + i * width, vals, width, label=SYSTEMS[key],
                   color=COLORS[key], alpha=0.85)

        ax.set_xticks(x + width * (len(systems) - 1) / 2)
        ax.set_xticklabels(all_domains, rotation=15, ha="right")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"OlympiadBench: {label}")
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=7)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plasticity_stability_by_domain.png"), dpi=150)
    plt.close()
    print("  Saved: plasticity_stability_by_domain.png")


# ── New Plots ─────────────────────────────────────────────────────────────────

def plot_bank_size_evolution(logs, output_dir):
    """Bank size over episodes for all memory-based systems."""
    fig, ax = plt.subplots(figsize=(12, 4))

    memory_systems = ["passive", "ism"]
    ism_tasks  = [ep["task"] for ep in logs["ism"]]
    boundaries = []
    prev_task  = ism_tasks[0]
    for i, t in enumerate(ism_tasks):
        if t != prev_task:
            boundaries.append(i)
            prev_task = t
    for b in boundaries:
        ax.axvline(b, color="gray", linestyle="--", alpha=0.35)

    prev_b    = 0
    prev_task = ism_tasks[0]
    for b in boundaries + [len(ism_tasks)]:
        mid = (prev_b + b) / 2
        ax.text(mid, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1,
                prev_task, ha="center", fontsize=7,
                transform=ax.get_xaxis_transform())
        prev_b    = b
        prev_task = ism_tasks[b] if b < len(ism_tasks) else ""

    for key in memory_systems:
        sizes = [ep.get("bank_size", 0) for ep in logs[key]]
        lw    = 2.5 if key == "ism" else 1.5
        ax.plot(sizes, label=SYSTEMS[key], color=COLORS[key],
                linewidth=lw, alpha=0.9)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Schema Bank Size")
    ax.set_title("OlympiadBench: Schema Bank Size Evolution")
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bank_size_evolution.png"), dpi=150)
    plt.close()
    print("  Saved: bank_size_evolution.png")


def plot_cumulative_accuracy(logs, output_dir):
    """Two-panel: (1) cumulative accuracy all systems, (2) ISM per-episode contribution."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Panel 1 — cumulative accuracy all systems
    ax = axes[0]
    ism_tasks  = [ep["task"] for ep in logs["ism"]]
    boundaries = []
    prev_task  = ism_tasks[0]
    for i, t in enumerate(ism_tasks):
        if t != prev_task:
            boundaries.append(i)
            prev_task = t
    for b in boundaries:
        ax.axvline(b, color="gray", linestyle="--", alpha=0.3)

    for key, episodes in logs.items():
        corrects = [int(ep["correct"]) for ep in episodes]
        cumacc   = [sum(corrects[:i+1]) / (i+1) for i in range(len(corrects))]
        lw = 2.5 if key == "ism" else 1.2
        ax.plot(cumacc, label=SYSTEMS[key], color=COLORS[key],
                linewidth=lw, alpha=0.9)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Accuracy")
    ax.set_title("OlympiadBench: Cumulative Accuracy (All Systems)")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(linestyle="--", alpha=0.3)

    # Panel 2 — ISM correct episodes over time (bar per episode)
    ax2 = axes[1]
    ism_corrects = [int(ep["correct"]) for ep in logs["ism"]]
    colors_ep    = [COLORS["ism"] if c else "#d62728" for c in ism_corrects]
    ax2.bar(range(len(ism_corrects)), ism_corrects, color=colors_ep,
            alpha=0.7, width=1.0)
    for b in boundaries:
        ax2.axvline(b, color="gray", linestyle="--", alpha=0.4)

    cumacc_ism = [sum(ism_corrects[:i+1]) / (i+1) for i in range(len(ism_corrects))]
    ax2_twin = ax2.twinx()
    ax2_twin.plot(cumacc_ism, color="black", linewidth=1.5,
                  linestyle="--", label="Cumulative Acc")
    ax2_twin.set_ylabel("Cumulative Accuracy", fontsize=9)
    ax2_twin.set_ylim(0, 1.0)
    ax2_twin.legend(fontsize=8, loc="lower right")

    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Correct (1) / Incorrect (0)")
    ax2.set_title("OlympiadBench: ISM Per-Episode Correctness")
    ax2.set_ylim(-0.1, 1.3)
    ax2.grid(linestyle="--", alpha=0.3)

    plt.suptitle("OlympiadBench: Cumulative Accuracy Analysis",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cumulative_accuracy.png"), dpi=150)
    plt.close()
    print("  Saved: cumulative_accuracy.png")


def plot_improvement_timeline(ism_analysis, output_dir):
    """Self-improvement events over episodes with bank size overlay."""
    if not ism_analysis:
        return

    improvement_log = ism_analysis.get("improvement_log", [])
    if not improvement_log:
        print("  [WARN] improvement_log empty — skipping improvement_timeline")
        return

    mechanism_colors = {
        "Self-Audit":        "#2196F3",
        "Self-Correct":      "#F44336",
        "Self-Merge":        "#9C27B0",
        "Self-Promote/Demote": "#FF9800",
        "Self-Prune":        "#4CAF50",
    }
    mechanism_markers = {
        "Self-Audit":        "o",
        "Self-Correct":      "X",
        "Self-Merge":        "D",
        "Self-Promote/Demote": "^",
        "Self-Prune":        "s",
    }

    # Extract audit bank sizes for the overlay line
    audit_eps   = []
    audit_sizes = []
    for entry in improvement_log:
        if entry.get("mechanism") == "Self-Audit":
            audit_eps.append(entry["episode"])
            audit_sizes.append(entry.get("bank_size", 0))

    fig, ax = plt.subplots(figsize=(13, 5))

    # Bank size line
    if audit_eps:
        ax.plot(audit_eps, audit_sizes, color="black", linewidth=1.5,
                linestyle="--", alpha=0.5, label="Bank size (at audit)")
        ax.fill_between(audit_eps, audit_sizes, alpha=0.06, color="black")

    # Scatter each mechanism
    seen = set()
    for entry in improvement_log:
        mech = entry.get("mechanism", "Unknown")
        ep   = entry.get("episode", 0)
        size = entry.get("bank_size_after",
               entry.get("bank_size", None))
        if size is None:
            continue
        col = mechanism_colors.get(mech, "#888888")
        mk  = mechanism_markers.get(mech, "o")
        label = mech if mech not in seen else None
        seen.add(mech)
        ax.scatter(ep, size, color=col, marker=mk, s=90,
                   zorder=4, label=label)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Schema Bank Size")
    ax.set_title("OlympiadBench: ISM Self-Improvement Timeline")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "improvement_timeline.png"), dpi=150)
    plt.close()
    print("  Saved: improvement_timeline.png")


def plot_ism_schema_usage(ism_analysis, output_dir):
    """Horizontal bar chart: synthesised schemas by usage count, shaded by SR."""
    if not ism_analysis:
        return

    bank = ism_analysis.get("bank_snapshot", [])
    synth = [s for s in bank if not s.get("is_seed", False)]
    if not synth:
        print("  [WARN] No synthesised schemas found — skipping ism_schema_usage")
        return

    synth.sort(key=lambda x: x.get("usage_count", 0), reverse=True)

    names  = [s["name"][:35] + "…" if len(s["name"]) > 35 else s["name"]
              for s in synth]
    usages = [s.get("usage_count", 0) for s in synth]
    srs    = [s.get("success_rate", 0.5) for s in synth]

    fig, ax = plt.subplots(figsize=(10, max(4, len(synth) * 0.45)))
    cmap    = plt.get_cmap("RdYlGn")
    norm    = plt.Normalize(0.3, 1.0)

    bars = ax.barh(range(len(names)), usages, color=[cmap(norm(r)) for r in srs],
                   edgecolor="white", alpha=0.9)

    for i, (bar, u, sr) in enumerate(zip(bars, usages, srs)):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{u} uses  SR={sr:.2f}", va="center", fontsize=8)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Usage Count")
    ax.set_title("OlympiadBench: ISM Synthesised Schema Usage\n(colour = success rate: green=high, red=low)")
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.6, label="Success Rate")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ism_schema_usage.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  Saved: ism_schema_usage.png")


# ── Save JSON Report ──────────────────────────────────────────────────────────

def save_report(domain_results, advantage_summary, transition_results, hard_domains, output_dir):
    report = {
        "dataset": "OlympiadBench",
        "accuracy_by_domain": {
            key: {
                domain: {"accuracy": round(v["acc"], 4), "correct": v["correct"], "total": v["total"]}
                for domain, v in domain_results[key].items()
            }
            for key in SYSTEMS
        },
        "hardest_domains": [
            {"domain": d, "avg_error_rate": round(e, 4)} for d, e in hard_domains
        ],
        "ism_advantage_over_baselines": advantage_summary,
        "block_transition_analysis": {
            key: [
                {k: (round(v, 4) if isinstance(v, float) else v) for k, v in b.items()}
                for b in blocks
            ]
            for key, blocks in transition_results.items()
        },
    }
    path = os.path.join(output_dir, "error_analysis_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved: error_analysis_report.json")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Error Analysis — OlympiadBench")
    print("=" * 60)

    logs = load_logs()
    print(f"Loaded {len(logs)} systems, {len(logs['ism'])} episodes each\n")

    # 1. Error patterns by domain
    print("[1] Computing error patterns by domain...")
    domain_results = error_patterns_by_domain(logs)
    for key in SYSTEMS:
        print(f"  {SYSTEMS[key]:<30}", end="")
        for d, v in sorted(domain_results[key].items()):
            print(f"  {d}: {v['acc']:.3f}", end="")
        print()

    # 2. Rolling accuracy
    print("\n[2] Computing rolling accuracy...")
    rolling_results = error_patterns_by_block(logs)

    # 3. Block transition
    print("\n[3] Computing block transition analysis...")
    transition_results = domain_transition_analysis(logs)

    # 4. Hardest domains (OlympiadBench specific)
    print("\n[4] Identifying hardest domains...")
    hard_domains = hardest_domains(domain_results)
    for d, e in hard_domains:
        print(f"  {d:<30} avg error rate: {e:.3f}")

    # 5. ISM advantage
    print("\n[5] Computing ISM advantage over baselines...")
    advantage   = compute_ism_advantage(logs)
    adv_summary = summarize_advantage(advantage)
    for row in adv_summary:
        print(f"  vs {row['baseline']:<30} "
              f"ISM wins: {row['ISM_wins']:>3}  "
              f"ISM loses: {row['ISM_loses']:>3}  "
              f"Net: {row['net_advantage']:>+4}  "
              f"Advantage rate: {row['advantage_rate']:.3f}")

    # 6. Load ISM analysis for ISM-specific plots
    print("\n[6] Loading ISM analysis...")
    ism_analysis = load_ism_analysis()

    # 7. Plots
    print("\n[7] Generating plots...")
    plot_accuracy_by_domain(domain_results, OUTPUT_DIR)
    plot_rolling_accuracy(rolling_results, logs, OUTPUT_DIR)
    plot_ism_advantage_bars(advantage, OUTPUT_DIR)
    plot_ism_advantage_by_domain(advantage, OUTPUT_DIR)
    plot_error_heatmap(domain_results, OUTPUT_DIR)
    plot_hardest_domains(domain_results, OUTPUT_DIR)
    plot_block_plasticity_stability(transition_results, OUTPUT_DIR)
    plot_bank_size_evolution(logs, OUTPUT_DIR)
    plot_cumulative_accuracy(logs, OUTPUT_DIR)
    plot_improvement_timeline(ism_analysis, OUTPUT_DIR)
    plot_ism_schema_usage(ism_analysis, OUTPUT_DIR)

    # 8. Save report
    print("\n[8] Saving JSON report...")
    save_report(domain_results, adv_summary, transition_results, hard_domains, OUTPUT_DIR)

    print(f"\nDone. All outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
