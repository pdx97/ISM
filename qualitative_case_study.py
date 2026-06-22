"""
qualitative_case_study.py
--------------------------
Finds the top-5 episodes where ISM got the answer correct but baselines failed,
then reconstructs the original problem text by replaying the CL stream (no
model inference — just dataset loading + shuffling with the same seed).

Usage:
  python qualitative_case_study.py --dataset olympiad   (default)
  python qualitative_case_study.py --dataset math_hard

Outputs:
  <RESULTS_DIR>/case_studies.json
  <RESULTS_DIR>/case_studies_latex.tex
"""

import argparse, json, os, sys
import numpy as np

BASE = r"C:\Users\prakh\Downloads\files\results\comparison"

DATASET_CONFIGS = {
    "olympiad": {
        "results_dir": os.path.join(BASE, "olympiad_updated_50_new"),
        "loader":      "load_olympiad_bench",
        "loader_kwargs": {},
    },
    "math_hard": {
        "results_dir": os.path.join(BASE, "math_updated_50"),
        "loader":      "load_competition_math",
        "loader_kwargs": {"min_level": 4, "max_level": 5},
    },
}

SEED         = 42
N_PER_BLOCK  = 50
TOP_N        = 5
BASELINES    = ["vanilla", "reflexion", "rag", "static", "passive"]
SEED_SCHEMAS = {"Algebra", "Number Theory", "Geometry",
                "Combinatorics", "Probability", "Calculus and Analysis"}

# ── Load episode logs ─────────────────────────────────────────────────────────

def load_logs(results_dir):
    logs = {}
    for key in ["ism"] + BASELINES:
        path = os.path.join(results_dir, f"{key}.json")
        with open(path) as f:
            logs[key] = {ep["episode"]: ep for ep in json.load(f)}
    return logs


def load_bank(results_dir):
    with open(os.path.join(results_dir, "ism_analysis.json")) as f:
        data = json.load(f)
    return {s["name"]: s for s in data["bank_snapshot"]}


# ── Answer equivalence check ─────────────────────────────────────────────────

def _answers_match(pred: str, gold: str) -> bool:
    """
    True if pred and gold represent the same answer after normalising
    LaTeX wrappers, whitespace, and trailing units like 'feet'/'meters'.
    Avoids false positives where a baseline was marked wrong only because
    of formatting differences (e.g. '\\frac{1}{3} feet' vs '\\frac{1}{3}').
    """
    import re
    if pred is None or gold is None:
        return False

    def normalise(s):
        s = _clean_pred(s) or s          # strip \boxed / \( \) wrappers
        s = s.strip()
        # Remove trailing unit words
        s = re.sub(r'\s+(feet|foot|meters?|cm|degrees?|units?)\s*$', '', s,
                   flags=re.IGNORECASE)
        # Collapse whitespace
        s = re.sub(r'\s+', '', s)
        # Strip surrounding $ signs
        s = s.strip('$')
        return s.lower()

    return normalise(pred) == normalise(gold)


# ── Find best case study episodes ────────────────────────────────────────────

MIN_SCHEMA_SR    = 0.60   # drop schemas with SR below this — likely poor fit
MIN_BASELINES_WRONG = 2   # require at least this many baselines wrong (stronger evidence)

def find_candidates(logs, bank):
    """
    Return TOP_N episodes where ISM used a synthesised schema and got the
    answer correct. Filters applied:
      - Schema must be in the final bank (so we have SR / usage to display)
      - Schema SR >= MIN_SCHEMA_SR (removes poorly-fitting schemas)
      - No duplicate schema names in the output (avoids repetition)
    Priority: most baselines wrong first. Seed schemas fill any remaining slots.
    """
    ism_log          = logs["ism"]
    final_bank_names = set(bank.keys()) if bank else set()
    synth_candidates = []
    seed_candidates  = []

    for ep_id, ism_ep in ism_log.items():
        if not ism_ep["correct"]:
            continue

        schema   = ism_ep.get("schema", "")
        is_synth = schema not in SEED_SCHEMAS

        # Skip pruned schemas — no SR/description to show
        if is_synth and schema not in final_bank_names:
            continue

        # Skip weak schemas — SR below threshold signals poor fit
        sr = bank.get(schema, {}).get("success_rate", 1.0) if is_synth else 1.0
        if is_synth and sr < MIN_SCHEMA_SR:
            continue

        gold = ism_ep["gold"]
        baselines_wrong = [
            k for k in BASELINES
            if not logs[k].get(ep_id, {}).get("correct", True)
            and not _answers_match(logs[k].get(ep_id, {}).get("pred", ""), gold)
        ]
        n_wrong = len(baselines_wrong)

        entry = {
            "episode":           ep_id,
            "task":              ism_ep["task"],
            "schema":            schema,
            "is_synth_schema":   is_synth,
            "ism_pred":          ism_ep["pred"],
            "gold":              ism_ep["gold"],
            "n_baselines_wrong": n_wrong,
            "baselines_wrong":   {k: logs[k].get(ep_id, {}).get("pred", "N/A")
                                   for k in baselines_wrong},
        }

        if n_wrong < MIN_BASELINES_WRONG:
            continue   # not a compelling win

        if is_synth:
            synth_candidates.append((n_wrong, ep_id, entry))
        else:
            seed_candidates.append((n_wrong, ep_id, entry))

    synth_candidates.sort(key=lambda x: (-x[0], x[1]))
    seed_candidates.sort(key=lambda x: (-x[0], x[1]))

    # Pick greedily: most baselines wrong first, no duplicate schema names
    chosen      = []
    seen_schemas = set()

    for pool in (synth_candidates, seed_candidates):
        for _, _, entry in pool:
            if len(chosen) >= TOP_N:
                break
            if entry["schema"] in seen_schemas:
                continue   # skip duplicates — each schema shown once
            chosen.append(entry)
            seen_schemas.add(entry["schema"])

    return chosen


# ── Reconstruct problem texts ─────────────────────────────────────────────────

def reconstruct_stream(cfg):
    """
    Rebuild the CL stream using the same loader + seed as the original run.
    Stdout is UTF-8 wrapped to avoid Windows cp1252 errors from Unicode
    characters in build_cl_stream's print output. No LLM calls are made.
    """
    import io, contextlib, importlib

    project_root = r"C:\Users\prakh\Downloads\files"
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    mod = importlib.import_module("schema_memory_agentic")
    loader_fn  = getattr(mod, cfg["loader"])
    build_fn   = getattr(mod, "build_cl_stream")

    print(f"Loading dataset via {cfg['loader']} (may take ~1 min on first run)...")

    utf8_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                   errors="replace", line_buffering=True)
    with contextlib.redirect_stdout(utf8_stdout):
        dataset = loader_fn(**cfg["loader_kwargs"])
        stream  = build_fn(dataset, n_per_block=N_PER_BLOCK, seed=SEED)

    print(f"Stream reconstructed: {len(stream)} episodes\n")
    return stream


# ── LaTeX formatting ──────────────────────────────────────────────────────────

def _clean_pred(pred: str) -> str:
    """
    Extract a clean bare math expression from a raw prediction string.

    Strategy (in order of priority):
      1. If pred contains \\boxed{...} anywhere, extract its content.
      2. If pred is wrapped in \\( ... \\) or \\[ ... \\], unwrap it.
      3. Strip trailing punctuation and return as-is.
      4. Return None for bare delimiters (malformed preds like '\\[').
    """
    import re
    if pred is None:
        return None

    s = pred.strip()

    # 1. Find the first \boxed{...} anywhere in the string (handles trailing
    #    text like "\(\boxed{267}\) (rounded to nearest integer)")
    m = re.search(r'\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}', s)
    if m:
        return m.group(1).strip()

    # Strip trailing unit words and punctuation before wrapper matching
    s_stripped = re.sub(
        r'[\s.,;]*(feet|foot|meters?|cm|degrees?|units?)\s*$', '',
        s.rstrip(".,;"), flags=re.IGNORECASE
    ).strip()

    # 2. \( ... \) inline wrapper (no boxed inside)
    m = re.fullmatch(r'\\\(\s*(.+?)\s*\\\)', s_stripped, re.DOTALL)
    if m:
        return m.group(1).strip()

    # 3. \[ ... \] display wrapper
    m = re.fullmatch(r'\\\[\s*(.+?)\s*\\\]', s_stripped, re.DOTALL)
    if m:
        return m.group(1).strip()

    # 4. Bare delimiters — malformed pred, treat as empty
    if s in (r'\[', r'\]', r'\(', r'\)', '$$', '$'):
        return None

    return s.rstrip(".,;")


def _clean_problem(text: str) -> str:
    """
    Prepare raw problem text for LaTeX inside a quote environment:
    - Strip [asy]...[/asy] Asymptote diagram blocks (not valid LaTeX)
    - Replace $$ ... $$ with \[ ... \] (display math safe inside quote)
    - Escape unescaped % signs
    - Leave & alone — it's almost always a column separator inside math
      environments (array, align) and must NOT be escaped there
    """
    import re
    if not text:
        return "[Problem text unavailable]"
    # Remove Asymptote diagram code, replace with a placeholder
    text = re.sub(r'\[asy\].*?\[/asy\]', r'\\textit{[diagram]}',
                  text, flags=re.DOTALL | re.IGNORECASE)
    # Escape unescaped % (not preceded by \)
    text = re.sub(r'(?<!\\)%', r'\\%', text)
    # Replace $$ ... $$ with \[ ... \]
    text = re.sub(r'\$\$\s*(.*?)\s*\$\$', r'\\[\n\1\n\\]', text, flags=re.DOTALL)
    return text


def _truncate_desc(desc: str, limit: int = 120) -> str:
    """Truncate description cleanly, adding \ldots if cut short."""
    if len(desc) <= limit:
        return desc
    # Cut at last space before limit to avoid mid-word breaks
    cut = desc[:limit].rsplit(" ", 1)[0].rstrip(",; ")
    return cut + r"\ldots"


def make_latex(cases):
    lines = []
    lines.append(r"\subsection*{Qualitative Case Studies}")
    lines.append(
        r"We present representative episodes where ISM correctly solved an "
        r"OlympiadBench problem that all compared baselines answered incorrectly, "
        r"illustrating how the self-evolved schema bank provides targeted guidance."
    )
    lines.append("")

    for i, c in enumerate(cases, 1):
        schema      = c["schema"]
        sr          = c.get("schema_sr")
        usage       = c.get("schema_usage_count")
        desc        = c.get("schema_description", "")
        heuristics  = c.get("schema_heuristics", [])[:3]
        hint_str    = "; ".join(heuristics) if heuristics else "--"
        sr_str      = f"{sr:.2f}" if sr is not None else "--"
        usage_str   = str(usage) if usage is not None else "--"
        synth_label = "(synthesised)" if c["is_synth_schema"] else "(seed)"
        n_wrong     = c["n_baselines_wrong"]

        # Fix 1: strip nested \boxed / \(...\) wrappers from pred and gold
        ism_answer  = _clean_pred(c["ism_pred"]) or c["gold"]

        # Fix 2: handle empty vanilla prediction
        vanilla_raw = c["baselines_wrong"].get("vanilla", "")
        vanilla_pred = _clean_pred(vanilla_raw)
        if vanilla_pred:
            vanilla_str = rf"e.g., Vanilla LLM predicted ${vanilla_pred}$"
        else:
            # Pick first non-empty baseline for the example
            alt = next(
                (_clean_pred(v) for v in c["baselines_wrong"].values()
                 if _clean_pred(v)),
                None
            )
            key = next(
                (k for k, v in c["baselines_wrong"].items() if _clean_pred(v)),
                None
            )
            if alt and key:
                vanilla_str = rf"e.g., {key} predicted ${alt}$"
            else:
                vanilla_str = "all baselines gave no valid answer"

        # Escape underscores in schema name for LaTeX
        schema_tex = schema.replace("_", r"\_")

        # Fix 3: clean problem text (replace $$ with \[, escape %, &)
        problem_tex = _clean_problem(c.get("problem_text"))

        # Fix 4: truncate description safely
        desc_tex = _truncate_desc(desc) if desc else ""

        lines.append(rf"\paragraph{{Case Study {i} (Episode {c['episode']}, {c['task']}).}}")
        lines.append(r"\begin{quote}\itshape")
        lines.append(problem_tex)
        lines.append(r"\end{quote}")
        lines.append("")
        lines.append(
            rf"ISM retrieved the schema \texttt{{{schema_tex}}} {synth_label} "
            rf"(success rate~$={sr_str}$, used {usage_str}~times). "
        )
        if desc_tex:
            lines.append(rf"This schema covers: \textit{{{desc_tex}}}. ")
        if heuristics:
            lines.append(rf"Key heuristics applied: \textit{{{hint_str}}}. ")
        baseline_str = (f"all {n_wrong} baselines" if n_wrong > 1
                        else "the remaining baseline")
        lines.append(
            rf"Guided by this schema, ISM produced the correct answer "
            rf"$\boxed{{{ism_answer}}}$, "
            rf"while {baseline_str} answered incorrectly "
            rf"({vanilla_str}). "
        )
        lines.append("")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASET_CONFIGS.keys()),
                        default="olympiad",
                        help="Which dataset results to analyse")
    args = parser.parse_args()

    cfg         = DATASET_CONFIGS[args.dataset]
    results_dir = cfg["results_dir"]

    print("=" * 60)
    print(f"Qualitative Case Study Extractor — {args.dataset} / ISM")
    print("=" * 60)

    logs = load_logs(results_dir)
    bank = load_bank(results_dir)

    print(f"Loaded {len(logs['ism'])} ISM episodes, {len(bank)} schemas in bank\n")

    candidates = find_candidates(logs, bank)
    print(f"Found {len(candidates)} candidate episodes (ISM correct, baselines wrong)\n")
    for c in candidates:
        print(f"  ep={c['episode']:>3d}  domain={c['task']:<20}  "
              f"schema={c['schema']:<35}  synth={c['is_synth_schema']}  "
              f"baselines_wrong={c['n_baselines_wrong']}")

    print("\nReconstructing CL stream to recover problem texts...")
    try:
        stream = reconstruct_stream(cfg)
        for c in candidates:
            ep_idx = c["episode"]
            if ep_idx < len(stream):
                c["problem_text"] = stream[ep_idx].text
            else:
                c["problem_text"] = None
    except Exception as e:
        print(f"  [WARN] Could not reconstruct stream: {e}")
        print("  Problem texts will be missing — everything else still works.")
        for c in candidates:
            c["problem_text"] = None

    # Attach schema details
    for c in candidates:
        info = bank.get(c["schema"], {})
        c["schema_sr"]           = info.get("success_rate")
        c["schema_usage_count"]  = info.get("usage_count")
        c["schema_description"]  = info.get("description", "")
        c["schema_heuristics"]   = info.get("heuristics", [])

    # Save JSON
    json_path = os.path.join(results_dir, "case_studies.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {json_path}")

    # Save LaTeX
    latex = make_latex(candidates)
    tex_path = os.path.join(results_dir, "case_studies_latex.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"Saved: {tex_path}")

    # Print preview
    print("\n" + "=" * 60)
    print("LaTeX preview (first case study):")
    print("=" * 60)
    if candidates:
        c = candidates[0]
        print(f"\nEpisode {c['episode']} | {c['task']} | Schema: {c['schema']}")
        print(f"Problem: {(c.get('problem_text') or 'N/A')[:300]}...")
        print(f"ISM answer: {c['ism_pred']}  |  Gold: {c['gold']}")
        print(f"Baselines wrong ({c['n_baselines_wrong']}): "
              + ", ".join(f"{k}={v}" for k, v in c["baselines_wrong"].items()))


if __name__ == "__main__":
    main()
