"""
Schema Memory — Results Visualization
======================================
Generate key figures from experiment results.

Usage:
    python plot_results.py [--results_dir results] [--output_dir figures]

Requires: matplotlib, numpy
    pip install matplotlib numpy
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional, Tuple

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
except ImportError:
    print("Install matplotlib and numpy: pip install matplotlib numpy")
    raise


# ── Constants ─────────────────────────────────────────────────────────────────

BLOCK_LABELS = [
    "Algebra",
    "Number Theory",
    "Counting & Prob.",
    "Algebra (rep 1)",
    "Number Theory (harder)",
    "Algebra (rep 2)",
]

# Explicit name map — avoids .title() mangling "ISM" → "Ism", "LLM" → "Llm"
NAME_MAP = {
    "static_llm":    "Static LLM",
    "neural_memory": "Neural Memory",
    "schema_memory": "Schema Memory",
    "ism":           "ISM",
}

SYSTEM_COLORS = {
    "Static LLM":    "#2ecc71",
    "Neural Memory": "#3498db",
    "Schema Memory": "#9b59b6",
    "ISM":           "#e74c3c",
}

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       11,
    "axes.linewidth":  1.2,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":      150,
})


# ══════════════════════════════════════════════════════════════════════════════
# LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def load_logs(results_dir: str) -> Dict[str, List[dict]]:
    """Load per-system episode logs from results_dir."""
    results_dir = Path(results_dir)
    logs = {}
    for key in NAME_MAP:
        path = results_dir / f"{key}.json"
        if path.exists():
            with open(path) as f:
                logs[key] = json.load(f)
        else:
            print(f"  [load] Not found: {path}")
    return logs


def load_lift_log(results_dir: str) -> List[dict]:
    """
    Load ism_lift_log.json — the real per-schema lift values recorded
    by the ISM at each audit point.  Falls back to [] if not present.
    """
    path = Path(results_dir) / "ism_lift_log.json"
    if not path.exists():
        print(f"  [load] ism_lift_log.json not found in {results_dir} "
              f"— Fig 4 will be skipped.")
        return []
    with open(path) as f:
        return json.load(f)


def load_summary(results_dir: str) -> List[dict]:
    path = Path(results_dir) / "summary.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def infer_n_per_block(log: List[dict]) -> int:
    """Infer episodes per block from first task switch."""
    if not log:
        return 20
    first_task = log[0]["task"]
    for i, e in enumerate(log):
        if e["task"] != first_task:
            return i
    return len(log)


def get_block_accuracies(log: List[dict], n_per_block: int) -> List[Tuple[str, float]]:
    """Return (label, accuracy) for each block in order."""
    results = []
    for block_idx in range(len(BLOCK_LABELS)):
        start = block_idx * n_per_block
        block = log[start : start + n_per_block]
        if not block:
            break
        correct = sum(1 for e in block if e.get("correct", False))
        acc = correct / len(block)
        label = block[0].get("task", BLOCK_LABELS[block_idx])
        results.append((label, acc))
    return results


def get_algebra_visits(log: List[dict], n_per_block: int) -> List[Optional[float]]:
    """
    Return accuracy for the 3 Algebra blocks: indices 0, 3, 5.
    Returns None for any block not present in the log.
    """
    algebra_block_indices = [0, 3, 5]
    accs = []
    for bi in algebra_block_indices:
        start = bi * n_per_block
        block = log[start : start + n_per_block]
        if not block:
            print(f"  [Fig 2] Warning: Algebra block at index {bi} "
                  f"(ep {start}–{start+n_per_block-1}) not found in log")
            accs.append(None)
        else:
            correct = sum(1 for e in block if e.get("correct", False))
            accs.append(correct / len(block))
    return accs


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Accuracy per block (grouped bar chart, 4 systems × 6 blocks)
# ══════════════════════════════════════════════════════════════════════════════

def fig1_accuracy_per_block(logs: Dict[str, List[dict]], output_path: Path):
    if not logs:
        print("  [Fig 1] No logs found, skipping")
        return

    # Infer n_per_block from first available log
    n_per_block = 20
    for log in logs.values():
        if log:
            n_per_block = infer_n_per_block(log)
            break

    # Collect block accuracies per system
    system_keys  = list(NAME_MAP.keys())
    system_names = [NAME_MAP[k] for k in system_keys]
    block_accs   = {}

    for key in system_keys:
        log = logs.get(key, [])
        name = NAME_MAP[key]
        if not log:
            block_accs[name] = [0.0] * 6
            continue
        pairs = get_block_accuracies(log, n_per_block)
        accs  = [acc for _, acc in pairs]
        # Pad to 6 blocks if run was shorter
        accs += [0.0] * (6 - len(accs))
        block_accs[name] = accs[:6]

    n_blocks = 6
    x        = np.arange(n_blocks)
    width    = 0.19
    offsets  = [-1.5, -0.5, 0.5, 1.5]

    fig, ax = plt.subplots(figsize=(13, 5.5))

    for i, name in enumerate(system_names):
        accs  = block_accs[name]
        color = SYSTEM_COLORS[name]
        bars  = ax.bar(x + offsets[i] * width, accs, width,
                       label=name, color=color, alpha=0.88,
                       edgecolor="white", linewidth=0.8)
        # Value labels on bars
        for bar, v in zip(bars, accs):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.008,
                        f"{v:.0%}", ha="center", va="bottom",
                        fontsize=7.5, color=color, fontweight="bold")

    # Shade the two repeat (forgetting test) columns
    for repeat_x, repeat_label in [(3, "Repeat 1"), (5, "Repeat 2")]:
        ax.axvspan(repeat_x - 0.5, repeat_x + 0.5,
                   alpha=0.07, color="#e74c3c", zorder=0)
        ax.text(repeat_x, 0.89, repeat_label,
                ha="center", fontsize=8.5, color="#e74c3c",
                fontweight="bold", alpha=0.75)

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_xlabel("Block", fontsize=12)
    ax.set_title("Figure 1: Per-Block Accuracy — 4 Systems × 6 Blocks",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(BLOCK_LABELS, fontsize=9.5)
    ax.set_ylim(0.35, 0.98)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{y:.0%}")
    )
    ax.legend(loc="upper right", fontsize=10, framealpha=0.85)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Forgetting curve (Algebra accuracy across 3 visits)
# ══════════════════════════════════════════════════════════════════════════════

def fig2_forgetting_curve(logs: Dict[str, List[dict]], output_path: Path):
    if not logs:
        print("  [Fig 2] No logs found, skipping")
        return

    n_per_block = 20
    for log in logs.values():
        if log:
            n_per_block = infer_n_per_block(log)
            break

    visits       = [1, 2, 3]
    visit_labels = [
        f"Visit 1\n(ep 1–{n_per_block})",
        f"Visit 2\n(ep {3*n_per_block+1}–{4*n_per_block})",
        f"Visit 3\n(ep {5*n_per_block+1}–{6*n_per_block})",
    ]

    fig, ax = plt.subplots(figsize=(7, 5))

    for key in NAME_MAP:
        log  = logs.get(key, [])
        name = NAME_MAP[key]
        if not log:
            continue
        accs  = get_algebra_visits(log, n_per_block)
        color = SYSTEM_COLORS[name]
        lw    = 3.0 if name == "ISM" else 1.8
        ls    = "-"  if name == "ISM" else "--"
        zord  = 4    if name == "ISM" else 3

        # Filter out None values for plotting
        valid_v = [v for v, a in zip(visits, accs) if a is not None]
        valid_a = [a for a in accs if a is not None]

        ax.plot(valid_v, valid_a,
                marker="o", linewidth=lw, linestyle=ls,
                color=color, markersize=9,
                label=name, zorder=zord)

        # End-point annotation
        if valid_a:
            ax.annotate(f"{valid_a[-1]:.0%}",
                        xy=(valid_v[-1], valid_a[-1]),
                        xytext=(valid_v[-1] + 0.08, valid_a[-1]),
                        fontsize=9.5, color=color,
                        va="center", fontweight="bold")

    ax.set_xticks(visits)
    ax.set_xticklabels(visit_labels, fontsize=10)
    ax.set_ylabel("Algebra Accuracy", fontsize=12)
    ax.set_ylim(0.45, 0.92)
    ax.set_xlim(0.7, 3.55)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{y:.0%}")
    )
    ax.set_title("Figure 2: Forgetting Curve — Algebra over 3 Visits",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.85)
    ax.grid(axis="y", alpha=0.3, linestyle=":")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — ISM bank state evolution
# ══════════════════════════════════════════════════════════════════════════════

def fig3_bank_evolution(logs: Dict[str, List[dict]], output_path: Path):
    ism_log = logs.get("ism", [])
    if not ism_log:
        print("  [Fig 3] No ISM log, skipping")
        return

    episodes   = [e["episode"] for e in ism_log]
    bank_sizes = [e["bank_size"] for e in ism_log]
    color      = SYSTEM_COLORS["ISM"]

    # Detect prune events: bank_size drops vs previous episode
    prune_eps  = [episodes[i] for i in range(1, len(bank_sizes))
                  if bank_sizes[i] < bank_sizes[i-1]]
    # Detect correction events: schema count stays same but we know audit fired
    # (every 10 episodes). We mark audit episodes separately.
    n_per_block = 20
    for log in logs.values():
        if log:
            n_per_block = infer_n_per_block(log)
            break
    audit_eps = [ep for ep in episodes if ep > 0 and ep % 10 == 0]

    fig, ax = plt.subplots(figsize=(10, 4.5))

    ax.step(episodes, bank_sizes, where="post",
            color=color, linewidth=2.2, zorder=3)
    ax.fill_between(episodes, bank_sizes,
                    step="post", alpha=0.15, color=color)

    # Mark prune events
    for pep in prune_eps:
        idx = episodes.index(pep)
        ax.axvline(pep, color="#888", linewidth=1, linestyle=":", alpha=0.7, zorder=2)
        ax.annotate("✂ prune",
                    xy=(pep, bank_sizes[idx]),
                    xytext=(pep + 1, bank_sizes[idx] + 0.5),
                    fontsize=7.5, color="#666",
                    arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.8))

    # Shade blocks
    block_colors = ["#f0f4ff", "#fff8f0", "#f0fff4", "#f0f4ff", "#fff8f0", "#f0f4ff"]
    for bi in range(6):
        start = bi * n_per_block
        end   = start + n_per_block
        ax.axvspan(start, min(end, max(episodes)),
                   alpha=0.25, color=block_colors[bi], zorder=0)
        ax.text(start + n_per_block / 2, max(bank_sizes) * 1.05,
                BLOCK_LABELS[bi].replace(" (rep 1)", "\n(rep 1)")
                                .replace(" (rep 2)", "\n(rep 2)")
                                .replace(" (harder)", "\n(harder)"),
                ha="center", fontsize=7, color="#666")

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Bank Size (# schemas)", fontsize=12)
    ax.set_title("Figure 3: ISM Bank Evolution — Schemas Added, Corrected, Pruned",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(0, max(episodes))
    ax.set_ylim(0, max(bank_sizes) * 1.18 + 1)
    ax.grid(axis="y", alpha=0.3, linestyle=":")

    # Legend for annotations
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=color, lw=2, label="Bank size"),
        Line2D([0], [0], color="#888", lw=1, ls=":", label="Prune event"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9, framealpha=0.85)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Schema lift improvement across audit cycles
# Uses ism_lift_log.json — the real audit values, not a reconstruction
# ══════════════════════════════════════════════════════════════════════════════

def fig4_schema_lift(lift_log: List[dict], output_path: Path,
                     schema_name: str = "Algebra-Linear-Equations"):
    if not lift_log:
        print("  [Fig 4] No lift log found — run with lift logging enabled, skipping")
        return

    # Filter to the target schema only
    schema_entries = [e for e in lift_log if e["schema"] == schema_name]
    if not schema_entries:
        print(f"  [Fig 4] Schema '{schema_name}' not found in lift log")
        print(f"  Available schemas: {sorted({e['schema'] for e in lift_log})}")
        return

    episodes = [e["episode"] for e in schema_entries]
    lifts    = [e["lift"]    for e in schema_entries]
    healths  = [e["health"]  for e in schema_entries]

    # Color each point by health
    health_colors = {
        "strong":  "#27ae60",
        "neutral": "#f39c12",
        "weak":    "#e74c3c",
        "unused":  "#95a5a6",
    }
    point_colors = [health_colors.get(h, "#888") for h in healths]

    color = SYSTEM_COLORS["ISM"]

    fig, ax = plt.subplots(figsize=(9, 4.5))

    # Line
    ax.plot(episodes, lifts, color=color, linewidth=2, zorder=2, alpha=0.7)

    # Points colored by health classification
    for ep, lift, hc in zip(episodes, lifts, point_colors):
        ax.scatter(ep, lift, color=hc, s=70, zorder=4, edgecolors="white", linewidths=0.8)

    # Zero baseline
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.6, linewidth=1.2)

    # Shade above/below zero
    ax.fill_between(episodes, lifts, 0,
                    where=[l >= 0 for l in lifts],
                    alpha=0.10, color="#27ae60", interpolate=True)
    ax.fill_between(episodes, lifts, 0,
                    where=[l < 0 for l in lifts],
                    alpha=0.10, color="#e74c3c", interpolate=True)

    # Annotate first correction (lift was negative → turned positive)
    for i in range(1, len(lifts)):
        if lifts[i-1] < 0 and lifts[i] >= 0:
            ax.annotate("crossed zero\n(self-correct worked)",
                        xy=(episodes[i], lifts[i]),
                        xytext=(episodes[i] + 3, lifts[i] + 0.04),
                        fontsize=8.5, color="#27ae60",
                        arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1))
            break

    ax.set_xlabel("Episode (audit point)", fontsize=12)
    ax.set_ylabel("Lift  (precision − baseline)", fontsize=12)
    ax.set_title(f"Figure 4: {schema_name} — Lift Across Audit Cycles",
                 fontsize=13, fontweight="bold", pad=12)
    ax.grid(alpha=0.25, linestyle=":")

    # Health legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
               markersize=9, label=h.capitalize())
        for h, c in health_colors.items()
        if h in set(healths)
    ]
    ax.legend(handles=legend_elements, title="Schema health",
              loc="upper left", fontsize=9, framealpha=0.85)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate figures from experiment results")
    parser.add_argument("--results_dir", default="results",
                        help="Directory with .json results (default: results)")
    parser.add_argument("--output_dir",  default="figures",
                        help="Directory to save figures (default: figures)")
    parser.add_argument("--schema",      default="Algebra-Linear-Equations",
                        help="Schema name for Fig 4 lift plot")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        print(f"Results dir not found: {results_dir}")
        return

    print(f"Loading results from: {results_dir}")
    logs     = load_logs(str(results_dir))
    lift_log = load_lift_log(str(results_dir))
    print(f"Loaded logs for: {list(logs.keys())}")
    print(f"Lift log entries: {len(lift_log)}")
    print()

    fig1_accuracy_per_block(logs, output_dir / "fig1_accuracy_per_block.png")
    fig2_forgetting_curve  (logs, output_dir / "fig2_forgetting_curve.png")
    fig3_bank_evolution    (logs, output_dir / "fig3_bank_evolution.png")
    fig4_schema_lift       (lift_log, output_dir / "fig4_schema_lift.png",
                            schema_name=args.schema)

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()