"""
run_comparison.py — Full System Comparison Runner

Runs all systems on the same CL stream and reports a unified comparison table.

Systems compared:
  1. Vanilla LLM          — no memory at all
  2. Static Schema        — one fixed hand-crafted schema
  3. RAG-over-Examples    — retrieve similar past problems as few-shot
  4. Reflexion            — verbal reflection memory (Shinn et al. NeurIPS 2023)
  5. Passive Schema Memory — schema bank, no self-improvement (from main file)
  6. ISM (Active)         — full self-improving schema memory (from main file)

Usage:
  python run_comparison.py --n_per_block 30 --min_level 4 --max_level 5
  python run_comparison.py --systems vanilla,rag,ism --n_per_block 20
  python run_comparison.py --seeds 42,123,456 --n_per_block 30
"""

import argparse
import json
import os
import random
import sys
import numpy as np
from pathlib import Path

# ── Import main module ────────────────────────────────────────────────────────
try:
    from schema_memory_agentic import (
        OPENAI_API_KEY,
        OPENAI_EMBEDDING_MODEL,
        LLM_MODEL,
        load_competition_math,
        load_math500,
        load_olympiad_bench,
        build_cl_stream,
        hard_reset_call,
        agentic_solve,
        evaluate_answer,
        extract_final_answer,
        CLTracker,
        build_schema_memory,
        build_intelligent_schema_memory,
        gpt4o_synthesizer,
    )
except ImportError as e:
    print(f"[ERROR] Cannot import schema_memory_agentic.py: {e}")
    sys.exit(1)

# ── Import baselines ──────────────────────────────────────────────────────────
try:
    from baseline_vanilla       import VanillaLLM
    from baseline_static_schema import StaticSchemaBaseline
    from baseline_rag_examples  import RAGExamplesBaseline
    from baseline_reflexion     import ReflexionBaseline
except ImportError as e:
    print(f"[ERROR] Cannot import baseline: {e}")
    sys.exit(1)

from openai import OpenAI
import re


# ── Shared encoder (same for all systems — controls for embedding variation) ──

class OpenAIEmbedEncoder:
    def __init__(self, client, model):
        self.client = client
        self.model  = model
        self._cache = {}

    def encode(self, text, normalize_embeddings=True):
        text = text.strip().replace("\n", " ")[:8000]
        if text in self._cache:
            return self._cache[text]
        resp = self.client.embeddings.create(input=[text], model=self.model)
        v = np.array(resp.data[0].embedding, dtype=np.float32)
        if normalize_embeddings:
            v /= np.linalg.norm(v)
        self._cache[text] = v
        return v


# ── System registry ───────────────────────────────────────────────────────────

ALL_SYSTEM_KEYS = ["vanilla", "static", "rag", "reflexion", "passive", "ism"]

SYSTEM_LABELS = {
    "vanilla":  "Vanilla LLM",
    "static":   "Static Schema",
    "rag":      "RAG-over-Examples",
    "reflexion":"Reflexion",
    "passive":  "Passive Schema Memory",
    "ism":      "ISM (Active)",
}


def load_dataset_for_comparison(
    dataset:     str,
    min_level:   int = 4,
    max_level:   int = 5,
    numeric_only: bool = False,
) -> list:
    """
    Load problems for the requested dataset.

    Supported:
      competition_math — lighteval/MATH-Hard, filtered by level
      math500          — lighteval/MATH-Hard (500-problem subset)
      olympiad         — math-ai/olympiadbench (674 olympiad problems)
    """
    if dataset == "competition_math":
        return load_competition_math(
            None, max_level=max_level, min_level=min_level,
            numeric_only=numeric_only
        )
    elif dataset == "math500":
        return load_math500()
    elif dataset == "olympiad":
        return load_olympiad_bench()
    else:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Choose: competition_math | math500 | olympiad"
        )


def build_systems(keys: list[str], encoder, synthesizer, verbose: bool) -> dict:
    systems = {}
    for key in keys:
        if key == "vanilla":
            systems[key] = VanillaLLM(encoder, verbose=False)
        elif key == "static":
            systems[key] = StaticSchemaBaseline(encoder, verbose=False)
        elif key == "rag":
            systems[key] = RAGExamplesBaseline(encoder, verbose=False)
        elif key == "reflexion":
            systems[key] = ReflexionBaseline(encoder, verbose=verbose)
        elif key == "passive":
            systems[key] = build_schema_memory(
                encoder, synthesizer=synthesizer, verbose=False
            )
        elif key == "ism":
            systems[key] = build_intelligent_schema_memory(
                encoder, synthesizer=synthesizer, verbose=verbose
            )
    return systems


# ── Solve dispatcher — handles custom solve paths ─────────────────────────────

def solve_problem(system_key: str, system, problem_text: str,
                  schema: dict, model: str) -> str:
    """
    Route to the right solve function depending on system type.
    RAG and Reflexion have custom solve() methods.
    All others use hard_reset_call.
    """
    if system_key in ("rag", "reflexion") and hasattr(system, "solve"):
        return system.solve(problem_text, schema)
    elif system_key == "vanilla" and hasattr(system, "solve"):
        return system.solve(problem_text)
    else:
        return hard_reset_call(problem_text, schema, model=model)


# ── Main comparison runner ────────────────────────────────────────────────────

def run_comparison(
    system_keys:  list[str] = None,
    dataset:      str = "competition_math",
    n_per_block:  int = 30,
    min_level:    int = 4,
    max_level:    int = 5,
    numeric_only: bool = False,
    model:        str = LLM_MODEL,
    output_dir:   str = "results/comparison",
    seed:         int = 42,
    verbose:      bool = False,
) -> dict:

    if system_keys is None:
        system_keys = ALL_SYSTEM_KEYS

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)

    # ── Encoder ───────────────────────────────────────────────────────────────
    print("Loading encoder...")
    _oai = OpenAI(api_key=OPENAI_API_KEY)
    encoder = OpenAIEmbedEncoder(_oai, OPENAI_EMBEDDING_MODEL)
    synthesizer = gpt4o_synthesizer

    # ── Systems ───────────────────────────────────────────────────────────────
    print(f"Building systems: {system_keys}")
    systems = build_systems(system_keys, encoder, synthesizer, verbose)

    # ── Dataset ───────────────────────────────────────────────────────────────
    print(f"Loading dataset: {dataset}")
    problems = load_dataset_for_comparison(
        dataset, min_level=min_level,
        max_level=max_level, numeric_only=numeric_only
    )
    if not problems:
        print("No problems loaded.")
        return {}

    stream = build_cl_stream(problems, n_per_block=n_per_block, seed=seed)
    print(f"Stream: {len(stream)} episodes\n")

    # ── Trackers & logs ───────────────────────────────────────────────────────
    trackers = {k: CLTracker() for k in system_keys}
    logs     = {k: []         for k in system_keys}

    W = 70
    print("═" * W)
    print(f"  COMPARISON: {len(system_keys)} systems | {len(stream)} episodes")
    print(f"  Model: {model} ")
    print("═" * W)

    # ── Episode loop ──────────────────────────────────────────────────────────
    for i, problem in enumerate(stream):
        block_label = problem.task_id
        ep_results  = {}

        for key in system_keys:
            sys_obj = systems[key]
            schema, features, ret_score, ret_breakdown, ret_status = \
                sys_obj.get_schema_for_problem(problem.text)

            response = solve_problem(key, sys_obj, problem.text, schema, model)
            correct  = evaluate_answer(
                response, problem.boxed_answer or "", problem.numeric_answer
            )
            outcome  = ("correct"   if correct else
                        "generic"   if ret_status in ("generic", "empty_bank") else
                        "incorrect")

            # ── ISM: on failure, run agentic tool loop for richer learning signal ──
            # All systems solve identically (hard_reset_call). When ISM fails,
            # we additionally invoke agentic_solve so that tool traces (sympy_verify,
            # calculate, schema_lookup misses, etc.) can feed the self-improvement
            # mechanisms. Tools are a learning aid, not a solving advantage.
            tool_trace = None
            if key == "ism" and outcome == "incorrect":
                sys_obj.bank._replay_ref = sys_obj.replay  # expose replay to search_past_failures
                agent_out  = agentic_solve(
                    problem.text, schema,
                    sys_obj.bank, sys_obj.encoder,
                    model=model, max_turns=8, verbose=False,
                )
                tool_trace = agent_out.get("tool_trace", [])

            sys_obj.after_episode(
                problem.text, features, schema,
                ret_status, outcome, block_label,
                tool_trace=tool_trace,
                response_text=response,
            )
            trackers[key].record(block_label, correct)

            bank_size   = sys_obj.bank.size()
            pred        = extract_final_answer(response) or ""
            schema_name = schema.get("name", "") if isinstance(schema, dict) else ""

            logs[key].append({
                "episode":   i,
                "task":      block_label,
                "correct":   correct,
                "pred":      pred,
                "gold":      problem.boxed_answer or "",
                "schema":    schema_name,
                "bank_size": bank_size,
            })
            ep_results[key] = {"correct": correct, "pred": pred, "bank": bank_size}

        # Per-episode print
        parts = [f"  Ep {i+1:>3} | {block_label:<28}"]
        for key in system_keys:
            res = ep_results[key]
            sym = "✓" if res["correct"] else "✗"
            lbl = SYSTEM_LABELS[key][:8]
            gold_str = str(problem.boxed_answer or "?")[:8]
            pred_str = str(res["pred"])[:8]
            parts.append(f"{lbl}:{sym} (gold={gold_str} pred={pred_str})")
        print(" | ".join(parts))

        # Print mistake / insight for ISM and Passive after the summary line
        for key in ("ism", "passive"):
            if key not in system_keys:
                continue
            sys_obj = systems[key]
            if not sys_obj.replay.buffer:
                continue
            last_ep = sys_obj.replay.buffer[-1]
            label   = SYSTEM_LABELS[key][:14]
            if not ep_results[key]["correct"] and last_ep.mistake:
                print(f"    [{label}] ✗ mistake : {last_ep.mistake}")
            elif ep_results[key]["correct"] and last_ep.insight:
                print(f"    [{label}] ✓ insight : {last_ep.insight}")

        # Every 20 episodes print rolling summary
        if (i + 1) % 20 == 0:
            _print_rolling(system_keys, logs, i, block_label)

    # ── Save logs ─────────────────────────────────────────────────────────────
    for key in system_keys:
        safe = key.replace(" ", "_").replace("(", "").replace(")", "")
        path = os.path.join(output_dir, f"{safe}.json")
        with open(path, "w") as f:
            json.dump(logs[key], f, indent=2)

    # ── Save ISM bank snapshot + improvement log ───────────────────────────────
    if "ism" in system_keys and "ism" in systems:
        _save_ism_analysis(systems["ism"], output_dir)
    if "passive" in system_keys and "passive" in systems:
        _save_passive_bank_snapshot(systems["passive"], output_dir)

    # ── Final report ──────────────────────────────────────────────────────────
    results = _print_final_report(system_keys, trackers, logs, output_dir)
    return results


def _save_ism_analysis(ism_system, output_dir: str):
    """Save ISM schema bank snapshot and self-improvement event log."""
    seed_names = {
        "Algebra", "Number Theory", "Geometry",
        "Combinatorics", "Probability", "Calculus and Analysis"
    }

    bank      = ism_system.bank
    hooks     = bank.hooks
    schemas   = bank.schemas

    # Bank snapshot: one entry per schema
    bank_snapshot = []
    for name, hook in hooks.items():
        bank_snapshot.append({
            "name":             name,
            "is_seed":          name in seed_names,
            "usage_count":      hook.usage_count,
            "success_rate":     round(hook.success_rate, 4),
            "correction_count": hook.correction_count,
            "description":      schemas.get(name, {}).get("description", "")[:200],
            "heuristics":       schemas.get(name, {}).get("heuristics", []),
        })
    bank_snapshot.sort(key=lambda x: -x["usage_count"])

    # New schemas only (non-seed, synthesized during the run)
    new_schemas = [s for s in bank_snapshot if not s["is_seed"]]

    # Self-improvement event log
    improvement_log = getattr(ism_system, "improvement_log", [])

    # Health log (one snapshot per audit)
    health_log = getattr(ism_system, "_health_log", [])

    # Lift log (per-schema lift per audit)
    lift_log = getattr(ism_system, "_lift_log", [])

    # Summary stats
    mechanism_counts = {}
    for event in improvement_log:
        m = event["mechanism"]
        mechanism_counts[m] = mechanism_counts.get(m, 0) + 1

    ism_analysis = {
        "bank_snapshot":      bank_snapshot,
        "new_schemas":        new_schemas,
        "improvement_log":    improvement_log,
        "health_log":         health_log,
        "lift_log":           lift_log,
        "summary": {
            "total_schemas":    len(bank_snapshot),
            "seed_schemas":     len([s for s in bank_snapshot if s["is_seed"]]),
            "new_schemas":      len(new_schemas),
            "mechanism_counts": mechanism_counts,
            "total_events":     len(improvement_log),
        },
    }

    path = os.path.join(output_dir, "ism_analysis.json")
    with open(path, "w") as f:
        json.dump(ism_analysis, f, indent=2)
    print(f"[Saved] ISM analysis → {path}")
    print(f"  Bank: {len(bank_snapshot)} schemas "
          f"({len(new_schemas)} new, {len(bank_snapshot) - len(new_schemas)} seed)")
    print(f"  Self-improvement events: {len(improvement_log)}")
    for m, c in mechanism_counts.items():
        print(f"    {m}: {c}x")


def _save_passive_bank_snapshot(passive_system, output_dir: str):
    """Save Passive Schema Memory bank snapshot for comparison."""
    seed_names = {
        "Algebra", "Number Theory", "Geometry",
        "Combinatorics", "Probability", "Calculus and Analysis"
    }
    bank  = passive_system.bank
    hooks = bank.hooks

    bank_snapshot = []
    for name, hook in hooks.items():
        bank_snapshot.append({
            "name":         name,
            "is_seed":      name in seed_names,
            "usage_count":  hook.usage_count,
            "success_rate": round(hook.success_rate, 4),
        })
    bank_snapshot.sort(key=lambda x: -x["usage_count"])

    path = os.path.join(output_dir, "passive_bank_snapshot.json")
    with open(path, "w") as f:
        json.dump(bank_snapshot, f, indent=2)
    print(f"[Saved] Passive bank snapshot → {path}")


def _print_rolling(keys, logs, ep_idx, block_label):
    print(f"\n  ── Rolling 20-ep summary at ep {ep_idx+1} [{block_label}] ──")
    print(f"  {'System':<28} {'Acc':>6}  {'Bank':>5}")
    print(f"  {'─'*28} {'─'*6}  {'─'*5}")
    for key in keys:
        log    = logs[key]
        recent = log[-20:]
        acc    = sum(e["correct"] for e in recent) / len(recent)
        bank   = log[-1]["bank_size"]
        label  = SYSTEM_LABELS.get(key, key)
        print(f"  {label:<28} {acc:>6.1%}  {bank:>5}")
    print()


def _print_final_report(keys, trackers, logs, output_dir) -> dict:
    W = 70
    print(f"\n{'═'*W}")
    print("  FINAL COMPARISON REPORT")
    print(f"{'═'*W}\n")

    # Per-system metrics
    results = {}
    for key in keys:
        log   = logs[key]
        total = len(log)
        acc   = sum(e["correct"] for e in log) / total if total else 0
        final_bank = log[-1]["bank_size"] if log else 0
        label = SYSTEM_LABELS.get(key, key)
        m     = trackers[key].compute() if hasattr(trackers[key], "compute") else {}
        results[key] = {
            "label":        label,
            "final_acc":    round(acc, 4),
            "plasticity":   round(m.get("plasticity", 0), 4),
            "stability":    round(m.get("stability", 0), 4),
            "forgetting":   round(m.get("forgetting", 0), 4),
            "bwt":          round(m.get("backward_transfer", 0), 4),
            "final_bank":   final_bank,
        }

    # Print comparison table
    header = (f"  {'System':<28} {'Acc':>6}  {'Plast':>6}  "
              f"{'Stab':>6}  {'Forg':>6}  {'BWT':>6}  {'Bank':>5}")
    print(header)
    print("  " + "─" * (len(header) - 2))

    # Sort by final accuracy descending
    for key in sorted(keys, key=lambda k: -results[k]["final_acc"]):
        r = results[key]
        print(
            f"  {r['label']:<28} {r['final_acc']:>6.1%}  "
            f"{r['plasticity']:>6.1%}  {r['stability']:>6.1%}  "
            f"{r['forgetting']:>6.1%}  {r['bwt']:>+6.1%}  "
            f"{r['final_bank']:>5}"
        )

    print(f"\n{'═'*W}")

    # Save
    out_path = os.path.join(output_dir, "comparison_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] {out_path}")

    return results


# ── Multi-seed comparison ─────────────────────────────────────────────────────

def run_multi_seed_comparison(
    seeds:       list[int],
    system_keys: list[str] = None,
    dataset:     str = "competition_math",
    n_per_block: int = 30,
    min_level:   int = 4,
    max_level:   int = 5,
    output_dir:  str = "results/comparison",
) -> dict:
    if system_keys is None:
        system_keys = ALL_SYSTEM_KEYS

    all_results = {}
    for seed in seeds:
        print(f"\n{'─'*60}")
        print(f"  Seed {seed}")
        print(f"{'─'*60}")
        seed_dir = os.path.join(output_dir, f"seed{seed}")
        r = run_comparison(
            system_keys=system_keys,
            dataset=dataset,
            n_per_block=n_per_block,
            min_level=min_level,
            max_level=max_level,
            output_dir=seed_dir,
            seed=seed,
        )
        all_results[seed] = r

    # Aggregate
    print(f"\n{'═'*70}")
    print("  MULTI-SEED AGGREGATE")
    print(f"{'═'*70}")
    print(f"  {'System':<28} {'Acc mean':>9}  {'Acc std':>8}")
    print(f"  {'─'*28} {'─'*9}  {'─'*8}")

    agg = {}
    for key in system_keys:
        accs = [all_results[s][key]["final_acc"]
                for s in seeds if key in all_results.get(s, {})]
        if not accs:
            continue
        mean = np.mean(accs)
        std  = np.std(accs, ddof=1) if len(accs) > 1 else 0.0
        agg[key] = {"mean_acc": round(float(mean), 4),
                    "std_acc":  round(float(std), 4)}
        label = SYSTEM_LABELS.get(key, key)
        print(f"  {label:<28} {mean:>9.1%}  {std:>8.1%}")

    agg_path = os.path.join(output_dir, "multi_seed_comparison.json")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(agg_path, "w") as f:
        json.dump({"seeds": seeds, "aggregate": agg,
                   "per_seed": {str(s): all_results[s] for s in seeds}}, f, indent=2)
    print(f"\n[Saved] {agg_path}")
    return agg


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="ISM vs. Baselines Comparison Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--systems",      type=str, default=",".join(ALL_SYSTEM_KEYS),
                   help="Comma-separated: vanilla,static,rag,reflexion,passive,ism")
    p.add_argument("--dataset",      type=str, default="competition_math",
                   choices=["competition_math", "math500", "olympiad"],
                   help="Dataset to use")
    p.add_argument("--n_per_block",  type=int, default=30)
    p.add_argument("--min_level",    type=int, default=4,
                   help="Min difficulty level (competition_math only)")
    p.add_argument("--max_level",    type=int, default=5,
                   help="Max difficulty level (competition_math only)")
    p.add_argument("--numeric_only", action="store_true")
    p.add_argument("--seeds",        type=str, default="42",
                   help="Comma-separated seeds. Multiple → multi-seed run.")
    p.add_argument("--output_dir",   type=str, default="results/comparison")
    p.add_argument("--verbose",      action="store_true")
    args = p.parse_args()

    keys  = [s.strip() for s in args.systems.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    if len(seeds) == 1:
        run_comparison(
            system_keys=keys,
            dataset=args.dataset,
            n_per_block=args.n_per_block,
            min_level=args.min_level,
            max_level=args.max_level,
            numeric_only=args.numeric_only,
            output_dir=args.output_dir,
            seed=seeds[0],
            verbose=args.verbose,
        )
    else:
        run_multi_seed_comparison(
            seeds=seeds,
            system_keys=keys,
            n_per_block=args.n_per_block,
            min_level=args.min_level,
            max_level=args.max_level,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
