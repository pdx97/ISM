"""
Bank Health Trajectory (BHT) for ISM (ARISE)
---------------------------------------------
H(t) = (strong + 0.5 * neutral) / total_schemas   at each Self-Audit checkpoint

Shows how the self-improvement loop regulates bank health over the stream:
- early rise as schemas activate
- temporary dip when many new schemas are synthesised (start as weak/unused)
- recovery after Self-Prune removes dead schemas
"""

import json, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

RESULTS_DIR = r"C:\Users\prakh\Downloads\files\results\comparison\math_updated_50"
OUT_DIR     = os.path.join(RESULTS_DIR, "schema_metrics")
os.makedirs(OUT_DIR, exist_ok=True)

with open(os.path.join(RESULTS_DIR, "ism_analysis.json")) as f:
    ism_analysis = json.load(f)

# ── Extract BHT from Self-Audit entries (deduplicate same episode) ────────────
seen_eps = set()
points   = []
for entry in ism_analysis["improvement_log"]:
    if entry.get("mechanism") != "Self-Audit":
        continue
    ep = entry["episode"]
    if ep in seen_eps:
        continue          # skip duplicate ep=300
    seen_eps.add(ep)
    hc = entry["health_counts"]
    total = sum(hc.values())
    if total == 0:
        continue
    score = (hc.get("strong", 0) * 1.0 + hc.get("neutral", 0) * 0.5) / total
    points.append((ep, score, hc, total))

episodes = np.array([p[0] for p in points])
scores   = np.array([p[1] for p in points])

# Trend for the recovery phase only (ep >= 160)
mask_recovery = episodes >= 160
if mask_recovery.sum() >= 2:
    slope_r, intercept_r, r_r, p_r, _ = linregress(episodes[mask_recovery],
                                                      scores[mask_recovery])
else:
    slope_r, intercept_r, r_r, p_r = 0, 0, 0, 1

# Overall trend
slope, intercept, r, p_val, _ = linregress(episodes, scores)

print("Bank Health Trajectory -- ISM (ARISE)")
print(f"  Checkpoints       : {len(points)}")
print(f"  Overall slope     : {slope:+.5f} per episode  (R2={r**2:.3f}  p={p_val:.3f})")
print(f"  Recovery slope    : {slope_r:+.5f} per episode  (ep>=160, R2={r_r**2:.3f}  p={p_r:.3f})")
print()
for ep, sc, hc, total in points:
    print(f"  ep={ep:>3d}  H={sc:.3f}  "
          f"[strong={hc.get('strong',0)} neutral={hc.get('neutral',0)} "
          f"weak={hc.get('weak',0)} unused={hc.get('unused',0)} total={total}]")

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})

fig, ax = plt.subplots(figsize=(10, 5))

# Phase shading
ax.axvspan(episodes[0], 50,  alpha=0.07, color="#9E9E9E")
ax.axvspan(50,          150, alpha=0.07, color="#FF9800")
ax.axvspan(150,         episodes[-1], alpha=0.07, color="#4CAF50")

# Phase labels at top
for x_mid, txt, col in [(25, "Seed\nactivation", "#9E9E9E"),
                         (100, "Synthesis\nburst", "#E65100"),
                         (225, "Self-regulation\n& recovery", "#2E7D32")]:
    ax.text(x_mid, 0.97, txt, ha="center", va="top", fontsize=8,
            color=col, fontweight="bold", transform=ax.get_xaxis_transform())

# Health score line
ax.plot(episodes, scores, "o-", color="#2196F3", linewidth=2.2,
        markersize=6, label="H(t)  bank health score", zorder=4)

# Recovery trend line (ep >= 160)
if mask_recovery.sum() >= 2:
    x_rec = episodes[mask_recovery]
    ax.plot(x_rec, slope_r * x_rec + intercept_r, "--", color="#4CAF50",
            linewidth=1.8, zorder=3,
            label=f"Recovery trend  slope={slope_r:+.5f}/ep  (R2={r_r**2:.2f})")

# Self-Prune event markers
prune_eps = sorted(set(e["episode"] for e in ism_analysis["improvement_log"]
                       if e.get("mechanism") == "Self-Prune"))
first = True
for ep in prune_eps:
    ax.axvline(ep, color="#FF9800", linewidth=1.2, linestyle=":", alpha=0.9,
               label="Self-Prune" if first else None)
    first = False

ax.set_xlabel("Episode", fontsize=12)
ax.set_ylabel("H(t)  =  (strong + 0.5*neutral) / total_schemas", fontsize=11)
ax.set_title("Bank Health Trajectory -- ISM (ARISE)", fontweight="bold", pad=14)
ax.set_ylim(-0.02, 1.05)
ax.legend(fontsize=9, framealpha=0.9, loc="lower left")

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "bht.pdf"), dpi=200)
fig.savefig(os.path.join(OUT_DIR, "bht.png"), dpi=150)
plt.close()
print(f"\nSaved: {OUT_DIR}/bht.png  and  bht.pdf")
