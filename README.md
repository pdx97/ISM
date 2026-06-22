# ISM: Self-Improving Strategy Memory for Continual Mathematical Reasoning

Official codebase for the paper
**["ISM: Self-Improving Strategy Memory for Continual Mathematical Reasoning"](https://openreview.net/pdf?id=5JK3t0YI5Z)**
by Prakhar Dixit and Tim Oates, accepted at the
[ICML 2026 AI4Math Workshop](https://openreview.net/forum?id=5JK3t0YI5Z).

> We propose **Intelligent Schema Memory (ISM)**, a self-evolving
> memory-augmented system that improves mathematical reasoning for a frozen
> LLM under continual learning with hard episodic resets. ISM maintains a
> compact, self-refined bank of strategy schemas learned from both
> successful and failed episodes, with symbolic tools that check
> intermediate steps and certify answers. Without updating model
> parameters, ISM outperforms passive, retrieval, and reflection baselines
> on MATH-Hard and OlympiadBench, using **64% and 86% fewer schemas**
> respectively than the strongest passive baseline.

📄 Paper: <https://openreview.net/pdf?id=5JK3t0YI5Z> &nbsp;·&nbsp;
🔗 OpenReview: <https://openreview.net/forum?id=5JK3t0YI5Z> &nbsp;·&nbsp;
💻 Code: <https://github.com/pdx97/ISM>

---

## 📊 Results

### 🧮 MATH-Hard (Level 4–5, competition\_math)

| System | Accuracy | Plasticity | Stability | Forgetting | Bank Size |
|--------|----------|------------|-----------|------------|-----------|
| Vanilla LLM | 44.0% | 41.3% | 46.7% | 6% | — |
| Static Schema | 76.0% | 72.0% | 80.0% | 9% | 1 |
| RAG-over-Examples | 54.0% | 50.7% | 57.3% | 5% | 300 |
| Reflexion | 51.3% | 48.0% | 54.7% | 7% | 300 |
| Passive Schema Memory | 78.7% | 75.3% | 82.0% | 10% | 47 |
| 🥇 **ISM (Ours)** | **80.7%** | **76.7%** | **84.7%** | **7%** | **17** |

### 🏆 OlympiadBench (Maths-COMP + Maths-CEE subset)

| System | Accuracy | Plasticity | Stability | Forgetting | Bank Size |
|--------|----------|------------|-----------|------------|-----------|
| Vanilla LLM | 23.7% | 24.0% | 23.3% | 5% | — |
| Static Schema | 59.7% | 58.0% | 61.3% | 6% | 1 |
| RAG-over-Examples | 33.3% | 32.7% | 34.0% | 2% | 300 |
| Reflexion | 29.7% | 26.7% | 32.7% | 2% | 300 |
| Passive Schema Memory | 59.7% | 57.3% | 62.0% | 10% | 91 |
| 🥇 **ISM (Ours)** | **61.7%** | **59.3%** | **64.0%** | **3%** | **13** |

✨ ISM achieves the highest accuracy and stability on both benchmarks while using **64–86% fewer schemas** than memory-based baselines.

---

## 🧠 How It Works

### 🎯 Problem Setup
Problems arrive as a stream of 300 episodes partitioned into 6 blocks of 50 (one domain per block). **Hard episodic resets** are enforced — no conversational context or problem history is shared across episodes. The LLM remains frozen; the only cross-episode channel is the external schema memory.

### 🏗️ Architecture

```
📝 Problem Text
  ↓
🧩 Hybrid Feature Extractor      ← rule-based + LLM branch, merged by agreement score
  ↓
🔍 Two-Stage Schema Retrieval    ← operator filter → soft scoring over feature hooks
  ↓
🤖 Schema-Guided LLM Call        ← single-shot hard reset call (same for all systems)
  ↓
✅ Answer Evaluation             ← binary correctness signal
  ↓
🧠 Memory Controller             ← hook update, failure-triggered agentic analysis,
                                   schema synthesis, periodic self-improvement
```

### 🗂️ Schema Bank
Each schema `s_k` stores a name, description, solution template, and heuristics. Its feature hook `h_k` stores operator type, structural pattern, heuristic set, quantity signature, an EMA embedding centroid, and a success rate updated online.

### ✨ Five Self-Improvement Mechanisms
1. 🩺 **Self-Audit** — scores each schema by outcome lift over stream baseline; labels schemas strong / neutral / weak / unused
2. 🛠️ **Self-Correct** — rewrites weak schemas using their failure cases
3. 🔗 **Self-Merge** — consolidates semantically similar non-seed schemas into one
4. 📈 **Self-Promote/Demote** — adjusts retrieval priority based on success rate
5. 🪓 **Self-Prune** — removes schemas never retrieved after 25 episodes, or confirmed weak (≥5 uses, SR < 0.38)

### 🧪 Failure-Triggered Agentic Tools
On incorrect answers, ISM runs an 8-turn agentic loop that may call: `search_past_failures` (recency-weighted replay search), `calculate` (sandboxed arithmetic), `sympy_verify` (formal verification), `decompose`, `schema_lookup`, and `self_verify`. Tool traces feed richer learning signals into schema updates — they never alter the submitted answer.

---

## ⚙️ Installation

Tested with Python 3.9.13 on Windows 11 and Linux. Higher Python versions
(3.10–3.12) are expected to work; the code does not use any 3.10+-only syntax.

```bash
git clone https://github.com/pdx97/ISM.git
cd ISM
python -m venv .venv
. .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Set your OpenAI API key in either source — environment variable takes priority:

```bash
# Option A: environment variable (preferred)
export OPENAI_API_KEY=sk-...    # Windows PowerShell: $env:OPENAI_API_KEY = "sk-..."

# Option B: local file (gitignored)
cp api_keys.example.json api_keys.json
# then edit api_keys.json and paste your key in
```

**Models used (pinned snapshots for reproducibility):**

| Role | Model |
|---|---|
| Main solver (every episode) | `gpt-4.1-mini` |
| Schema synthesis, Self-Correct, Self-Merge, Self-Reinforce, Self-Antipattern | `gpt-4o` |
| Operator classification | `gpt-4o-mini` |
| Embeddings (problem + schema centroids) | `text-embedding-3-small` |

If OpenAI deprecates a snapshot, pin a dated alias (e.g.
`gpt-4.1-mini-2025-04-14`) on the corresponding constant in
`schema_memory_agentic.py` to keep results reproducible.

---

## Running Experiments

### Full system comparison (all 6 systems)
```bash
# MATH-Hard Level 4–5
python run_comparison.py --dataset competition_math --n_per_block 50 --min_level 4 --max_level 5

# OlympiadBench
python run_comparison.py --dataset olympiad --n_per_block 50
```

### ISM standalone
```bash
# Quick smoke run (20 problems/block, level 1-2 only)
python schema_memory_agentic.py --quick

# Full single-system run on MATH-500
python schema_memory_agentic.py --dataset math500 --n_per_block 50

# Focused ablation: Passive Schema Memory vs ISM (active)
python schema_memory_agentic.py --ablation --dataset math500 --n_per_block 40

# Inspect a saved run without re-executing
python schema_memory_agentic.py --inspect --output_dir results/math500/<run_name>
```

Common flags: `--seed 42`, `--no_synthesizer` (disable schema evolution),
`--agentic` (enable failure-triggered tool use for ISM), `--pass_k K`.

### Analysis and plots
```bash
# Error analysis with 11 plots (accuracy by domain, rolling accuracy, ISM advantage, etc.)
python error_analysis_olympiad.py
python error_analysis_math_updated.py

# Qualitative case studies — ISM wins using synthesised schemas
python qualitative_case_study.py --dataset math_hard
python qualitative_case_study.py --dataset olympiad

# Bank Health Trajectory (BHT) metric
python schema_metrics.py
```

### Tests
```bash
python smoke_test.py   # offline component checks (no API calls, < 5 s)
```

`smoke_test.py` exercises feature extraction, the schema bank, the replay
buffer, the answer evaluator, and CL metrics. Useful as a pre-commit
sanity check and to confirm imports resolve after dependency updates.

---

## Output Structure

```
results/comparison/{run_name}/
  ├── {system}.json          # per-episode log: episode, task, correct, pred, gold, schema, bank_size
  ├── ism_analysis.json      # bank snapshot + improvement log (all Self-* events)
  └── comparison_results.json

results/error_analysis/{run_name}/
  ├── accuracy_by_domain.png
  ├── rolling_accuracy.png
  ├── ism_advantage.png
  ├── bank_size_evolution.png
  ├── cumulative_accuracy.png
  ├── improvement_timeline.png
  └── ism_schema_usage.png
```

---

## Baselines

| System | Description |
|--------|-------------|
| Vanilla LLM | No memory; fresh prompt each episode |
| Static Schema | One hand-crafted schema always injected |
| RAG-over-Examples | Retrieves past solved problems as few-shot examples |
| Reflexion | Verbal reflection stored in free-text memory |
| Passive Schema Memory | Schema bank with no self-improvement |
| **ISM** | Full schema bank with 5 self-improvement mechanisms + agentic tools |

---

## Reproducibility

All numbers in the results tables above come from the runs saved in
`results/comparison/math_updated_50/` (MATH-Hard) and
`results/comparison/olympiad_updated_50_new/` (OlympiadBench).

Exact commands used:

```bash
# MATH-Hard (Level 4–5), 50 problems per block, fixed seed
python run_comparison.py \
    --dataset competition_math \
    --n_per_block 50 \
    --min_level 4 --max_level 5 \
    --seeds 42 \
    --output_dir results/comparison/math_updated_50

# OlympiadBench, 50 problems per block, fixed seed
python run_comparison.py \
    --dataset olympiad \
    --n_per_block 50 \
    --seeds 42 \
    --output_dir results/comparison/olympiad_updated_50_new
```

**Stream layout.** Both runs use the CL stream constructed by
`build_cl_stream` in `schema_memory_agentic.py`: six per-domain blocks
followed by two repeat blocks of the first task family (for the forgetting
test) and one harder-variant block.

**Determinism caveat.** The solver (`gpt-4.1-mini`) and embedding model are
deterministic with `temperature=0.0`. The schema synthesizer
(`gpt-4o` via `gpt4o_synthesizer`, `_ism_correct`, `_ism_merge`) currently
runs at OpenAI's default sampling temperature, so the *contents* of
synthesized schemas may vary between reruns; the bank size, retrieval
behavior, and final accuracy are stable to within ~1 episode in our
sweeps. To force fully deterministic synthesis, route the four direct
`client.chat.completions.create(model=SYNTHESIZER_MODEL, ...)` calls
through `_llm_kwargs(SYNTHESIZER_MODEL, ...)`.

**Approximate cost per run** (gpt-4.1-mini + gpt-4o synthesis, US-East,
prices as of 2026-Q2): ~$0.50–$1.00 per 300-episode comparison run for
MATH-Hard; ~$0.80–$1.50 for OlympiadBench (longer problems).

---

## Citation

If you use this software or its results, please cite the paper:

```bibtex
@inproceedings{dixit2026ism,
  title     = {ISM: Self-Improving Strategy Memory for Continual Mathematical Reasoning},
  author    = {Dixit, Prakhar and Oates, Tim},
  booktitle = {ICML 2026 AI4Math Workshop},
  year      = {2026},
  url       = {https://openreview.net/forum?id=5JK3t0YI5Z}
}
```

GitHub also offers automatic citation export via the
[`CITATION.cff`](./CITATION.cff) file in this repo.

---

## License

Code released under the [MIT License](./LICENSE).
The accompanying paper is released by the authors under CC BY 4.0 on OpenReview.
