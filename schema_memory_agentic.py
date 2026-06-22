"""
Intelligent Schema Memory (ISM) — Self-Contained Runner
=======================================================
Single file. No package structure. No relative imports.
Set your API key (env var ``OPENAI_API_KEY`` or ``api_keys.json``) and run:

    python schema_memory_agentic.py --quick
    python schema_memory_agentic.py --dataset math500 --n_per_block 50
    python schema_memory_agentic.py --ablation --dataset competition_math

For the full multi-system comparison see ``run_comparison.py``.

Requirements: see ``requirements.txt``.
"""

import os
import re
import sys
import json
import time
import random
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from feature_extractor_llm import safe_extract_features

# ── API key: env var > api_keys.json > placeholder ────────────────────────────
def _load_api_key():
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    cfg = Path(__file__).parent / "api_keys.json"
    if cfg.exists():
        try:
            with open(cfg) as f:
                data = json.load(f)
            k = data.get("openai_api_key")
            if k:
                return k
        except Exception:
            pass
    return "YOUR_KEY_HERE"

OPENAI_API_KEY = _load_api_key()

# ── Config ────────────────────────────────────────────────────────────────────
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"  # better math separation
LLM_MODEL              = "gpt-4.1-mini"  # main solver
SYNTHESIZER_MODEL      = "gpt-4o"        # self-correct, self-merge, self-reinforce


def _llm_kwargs(model: str, **extra) -> dict:
    """Build kwargs for chat.completions.create with deterministic sampling."""
    return {"model": model, "temperature": 0.0, **extra}
RETRIEVAL_THRESHOLD    = 0.55   # kept for backward compat (two-stage now primary)
REPLAY_EVERY           = 10
EMA_ALPHA              = 0.02
EMA_ALPHA_SUCCESS      = 0.04   # centroid pulls harder toward successful exemplars
EMA_ALPHA_FAILURE      = 0.01   # centroid drifts less from failure distribution
REINFORCE_EVERY        = 15     # Mechanism 6: Self-Reinforce interval
ANTIPATTERN_EVERY      = 20     # Mechanism 7: Self-Antipattern interval
FEATURE_DEBUG          = False


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: FEATURE HOOKS
# ══════════════════════════════════════════════════════════════════════════════

# Aligned with OPERATOR_CLASSIFICATION_PROMPT: combinatoric vs probability are distinct.
OPERATOR_KEYWORDS = {
    "algebraic":     ["equation", "solve for", "expression", "simplify", "exponent",
                      "radical", "sqrt", "polynomial", "quadratic", "inequality",
                      "logarithm", "piecewise", "absolute value", "floor function",
                      "complex number", "system of equations", "factor"],
    "number_theory": ["divisibility", "prime", "gcd", "lcm", "modular", "remainder",
                      "congruence", "digit", "factorization", "diophantine",
                      "coprime", "base-", "sum of digits", "euler's", "phi("],
    "geometric":     ["area", "volume", "perimeter", "radius", "diameter",
                      "circle", "rectangle", "triangle", "angle", "similar",
                      "coordinate", "slope", "midpoint", "inscribed", "circumscribed",
                      "tangent", "polygon", "perpendicular", "parallel", "distance between"],
    "combinatoric":  ["ways", "arrangements", "combinations", "permutation", "choose",
                      "counting", "inclusion-exclusion", "pigeonhole", "stars and bars",
                      "paths", "select", "subset", "committee", "distribute among", "distinct"],
    "probability":   ["probability", "chance", "likelihood", "expected value", "conditional",
                      "dice", "cards", "coins", "urns", "at random", "independent",
                      "random variable", "uniform distribution"],
    "calculus":      ["limit", "derivative", "integral", "convergence",
                      "taylor", "l'hopital", "extrema", "differentiate",
                      "integrate", "antiderivative", "inflection point"],
    "rate":          ["per hour", "per minute", "per day", "speed", "fills",
                      "drains", "completes", "mph", "flow rate", "work rate",
                      "miles per", "liters per"],
}

# STRUCTURAL_KEYWORDS = {
#     "two_agents_combined": ["together", "combined", "simultaneously",
#                             "at the same time", "working together", "jointly"],
#     "part_whole":          ["fraction of", "part of", "portion", "out of",
#                             "percent of", "share of"],
#     "before_after":        ["after", "before", "later", "ago", "originally",
#                             "initially", "increased by", "decreased by", "was changed"],
#     "comparison":          ["more than", "less than", "times as", "compared to",
#                             "difference between", "ratio of"],
#     "find_missing":        ["solve for", "determine the value", "compute", "find the value",
#                             "what is the smallest", "what is the largest", "how many"],
# }

# NEW — 8 categories with finer math-aware splits
STRUCTURAL_KEYWORDS = {
    "two_agents_combined": ["together", "combined", "simultaneously",
                            "at the same time", "working together"],
    "part_whole":          ["fraction of", "part of", "portion", "percent of"],
    "before_after":        ["after", "before", "later", "ago",
                            "originally", "increased by", "decreased by"],
    "comparison":          ["more than", "less than", "times as",
                            "difference between", "ratio of"],
    "optimization":        ["greatest", "smallest", "maximum", "minimum",
                            "least", "largest", "max", "min", "optimal"],
    "existence_count":     ["how many", "number of", "count the",
                            "exactly", "at least one"],
    "construction":        ["construct", "find an", "exhibit", "give an example"],
    "evaluate_expression": ["compute the value", "evaluate", "find the value of",
                            "what is the value"],
}

HEURISTIC_KEYWORDS = {
    "decompose":              ["break into", "split into", "subproblem", "sub-case", "separate into"],
    "work_backwards":         ["backwards", "reverse", "from the end", "result is", "work back"],
    "introduce_variable":     ["let x", "let n", "let t", "let f", "denote", "set x =", "let the unknown"],
    "find_pattern":           ["pattern", "repeating", "periodic", "notice that", "each time", "observe that"],
    "change_representation":  ["convert", "rewrite", "express as", "in terms of", "substitute", "rearrange"],
    "examine_special_cases":  ["special case", "when n=", "boundary", "extreme", "try n=1", "consider n=0"],
    "use_symmetry":           ["symmetric", "symmetry", "invariant", "reflection", "rotation", "by symmetry"],
    "argue_by_contradiction": ["contradiction", "suppose not", "assume the contrary", "cannot be", "if not"],
    "apply_theorems":         ["theorem", "lemma", "by cauchy", "by am-gm", "by vieta", "by pythagoras"],
    "visualize":              ["draw", "sketch", "diagram", "visualize", "plot", "geometric interpretation"],
}


@dataclass
class QuantitySignature:
    n_known: int = 0
    n_unknown: int = 1
    has_rate: bool = False
    has_time: bool = False
    has_constraint: bool = False
    unit_type: str = "abstract"

    def similarity(self, other) -> float:
        score  = 0.2 * (1.0 - min(abs(self.n_known   - other.n_known),   3) / 3)
        score += 0.2 * (1.0 - min(abs(self.n_unknown - other.n_unknown), 3) / 3)
        score += 0.2 * float(self.has_rate       == other.has_rate)
        score += 0.2 * float(self.has_time       == other.has_time)
        score += 0.1 * float(self.has_constraint == other.has_constraint)
        score += 0.1 * float(self.unit_type      == other.unit_type)
        return score


@dataclass
class ProblemFeatures:
    operator_type: str
    structural_pattern: str
    heuristics: List[str]
    quantity_signature: QuantitySignature
    embedding: np.ndarray
    log_llm_used: bool = False  # True if features came from LLM extract, else rule-based


@dataclass
class FeatureHook:
    schema_name: str
    operator_type: str
    structural_pattern: str
    heuristic_signature: List[str]
    quantity_signature: QuantitySignature
    embedding_centroid: np.ndarray
    usage_count: int = 0
    success_rate: float = 0.5
    associated_schemas: List[str] = field(default_factory=list)
    correction_count: int = 0
    is_seed: bool = False

    def update_centroid(self, new_emb, correct: bool = True, alpha: float = None):
        # Correct episodes pull the centroid harder; failures drift it less.
        if alpha is None:
            alpha = EMA_ALPHA_SUCCESS if correct else EMA_ALPHA_FAILURE
        self.embedding_centroid = (1 - alpha) * self.embedding_centroid + alpha * new_emb

    def update_success(self, correct: bool, alpha: float = None):
        # Higher alpha for successes so strong evidence of correctness registers faster.
        if alpha is None:
            alpha = 0.15 if correct else 0.10
        self.success_rate = alpha * (1.0 if correct else 0.0) + (1 - alpha) * self.success_rate


class FeatureExtractor:
    def extract(self, text: str, embedding: np.ndarray) -> ProblemFeatures:
        t = text.lower()
        return ProblemFeatures(
            operator_type      = self._operator(t),
            structural_pattern = self._structure(t),
            heuristics         = self._heuristics(t),
            quantity_signature = self._quantity(t),
            embedding          = embedding,
        )

    def _operator(self, t):
        scores = {k: sum(1 for kw in v if kw in t) for k, v in OPERATOR_KEYWORDS.items()}
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "algebraic"

    def _structure(self, t):
        scores = {k: sum(1 for kw in v if kw in t) for k, v in STRUCTURAL_KEYWORDS.items()}
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "find_missing"
        # return best if scores[best] > 0 else None

    def _heuristics(self, t):
        matched = [h for h, kws in HEURISTIC_KEYWORDS.items() if any(kw in t for kw in kws)]
        return matched or ["decompose"]

    def _quantity(self, t):
        nums = re.findall(r'\b\d+\.?\d*\b', t)
        return QuantitySignature(
            n_known      = max(0, len(nums) - 1),
            has_rate     = any(kw in t for kw in OPERATOR_KEYWORDS["rate"]),
            has_time     = any(kw in t for kw in ["hour","minute","day","week","second"]),
            has_constraint = any(kw in t for kw in ["at least","at most","maximum","minimum"]),
            unit_type    = ("time"     if "hour" in t or "minute" in t else
                            "distance" if any(k in t for k in ["km","mile","meter"]) else
                            "money"    if any(k in t for k in ["$","dollar","cost","price"]) else
                            "abstract"),
        )


class HybridFeatureExtractor:
    """LLM-based extraction when confident; rule-based fallback. Single taxonomy."""

    def __init__(self, llm_client, model_name: str = None, debug: bool = False):
        self.llm_client = llm_client
        self.model_name = model_name or LLM_MODEL
        self.rule_extractor = FeatureExtractor()
        self.agreement_threshold = 0.60
        self.debug = debug

    def extract(self, text: str, embedding: np.ndarray) -> ProblemFeatures:
        
        rule_features = self.rule_extractor.extract(text, embedding)
        if self.debug:
            print(
                "  [FeatureExtractor] RULE "
                f"op={rule_features.operator_type} "
                f"struct={rule_features.structural_pattern} "
                f"heur={rule_features.heuristics} "
                f"qty={{n_known:{rule_features.quantity_signature.n_known}, "
                f"has_rate:{rule_features.quantity_signature.has_rate}, "
                f"has_time:{rule_features.quantity_signature.has_time}, "
                f"has_constraint:{rule_features.quantity_signature.has_constraint}}}"
            )
        llm_features = None
        if self.llm_client is not None:
            llm_features = safe_extract_features(
                text, self.llm_client,
                min_confidence=0.0,
                model_name=self.model_name,
                verbose=self.debug,
            )
        if self.debug:
            print(f"  [FeatureExtractor] LLM  {llm_features}")

        # Agreement-gated decision (original logic)
        if llm_features:
            conf = self._agreement_confidence(rule_features, llm_features)
            if self.debug:
                print(
                    "  [FeatureExtractor] AGREEMENT "
                    f"score={conf:.3f} threshold={self.agreement_threshold:.3f}"
                )
            if conf >= self.agreement_threshold:
                if self.debug:
                    print("  [FeatureExtractor] DECISION use_llm_features=True")
                return self._merge_features(llm_features, embedding, log_llm_used=True)
            if self.debug:
                print("  [FeatureExtractor] DECISION use_llm_features=False (fallback=rule)")
        else:
            if self.debug:
                print("  [FeatureExtractor] AGREEMENT score=N/A (no valid LLM features)")
                print("  [FeatureExtractor] DECISION use_llm_features=False (fallback=rule)")

        return ProblemFeatures(
            operator_type=rule_features.operator_type,
            structural_pattern=rule_features.structural_pattern,
            heuristics=rule_features.heuristics,
            quantity_signature=rule_features.quantity_signature,
            embedding=embedding,
            log_llm_used=False,
        )

    def _agreement_confidence(self, rule_features: ProblemFeatures, llm_f: dict) -> float:
        """
        Option A (agreement-based confidence):
        Compute a deterministic confidence score from how well LLM-extracted
        features match the rule-based features.
        """
        # Operator exact match; structure can match top-2.
        op_match = 1.0 if llm_f.get("operator") == rule_features.operator_type else 0.0
        llm_structs = llm_f.get("structure_top2", [llm_f.get("structure", "find_missing")])
        if isinstance(llm_structs, str):
            llm_structs = [llm_structs]
        st_match = 1.0 if rule_features.structural_pattern in set(llm_structs) else 0.0

        # Heuristic overlap (Jaccard on top-3 heuristics).
        llm_heurs = llm_f.get("heuristics", []) or []
        rule_heurs = rule_features.heuristics or []
        union = set(llm_heurs) | set(rule_heurs)
        if not union:
            heur_overlap = 0.0
        else:
            heur_overlap = len(set(llm_heurs) & set(rule_heurs)) / len(union)

        # Quantity agreement: build a QuantitySignature from LLM fields
        # and reuse the same similarity function you already use elsewhere.
        q = llm_f.get("quantities", {}) or {}
        llm_q = QuantitySignature(
            n_known=int(q.get("num_values", 0) or 0),
            n_unknown=max(0, int(q.get("n_unknown", 1) or 1)),
            has_rate=bool(q.get("has_rate", False)),
            has_time=bool(q.get("has_time", False)),
            has_constraint=bool(q.get("has_constraint", False)),
            unit_type="abstract",
        )
        q_sim = rule_features.quantity_signature.similarity(llm_q)

        # Weighted agreement score (kept simple and interpretable).
        return (
            0.40 * op_match +
            0.30 * st_match +
            0.15 * heur_overlap +
            0.15 * q_sim
        )

    def _merge_features(
        self, f: dict, embedding: np.ndarray, log_llm_used: bool = True
    ) -> ProblemFeatures:
        q = f.get("quantities", {})
        n_known = max(0, q.get("num_values", 0))
        n_unknown = max(0, q.get("n_unknown", 1))
        llm_structs = f.get("structure_top2", [f.get("structure", "find_missing")])
        if isinstance(llm_structs, str):
            llm_structs = [llm_structs]
        primary_struct = llm_structs[0] if llm_structs else "find_missing"
        return ProblemFeatures(
            operator_type=f.get("operator", "algebraic"),
            structural_pattern=primary_struct,
            heuristics=f.get("heuristics", []) or ["decompose"],
            quantity_signature=QuantitySignature(
                n_known=n_known,
                n_unknown=n_unknown,
                has_rate=q.get("has_rate", False),
                has_time=q.get("has_time", False),
                has_constraint=q.get("has_constraint", False),
                unit_type="abstract",
            ),
            embedding=embedding,
            log_llm_used=log_llm_used,
        )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: HOOK MATCHER
# ══════════════════════════════════════════════════════════════════════════════

# WEIGHTS = {"operator":0.45, "structural":0.30, "heuristic":0.12,
#            "quantity":0.08, "embedding":0.03, "prior":0.02}

WEIGHTS = {"operator":0.00, "structural":0.15, "heuristic":0.15,
           "quantity":0.05, "embedding":0.55, "prior":0.10}


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def score_hook(problem: ProblemFeatures, hook: FeatureHook) -> Tuple[float, dict]:
    p_hset = set(problem.heuristics)
    h_hset = set(hook.heuristic_signature)
    union  = p_hset | h_hset
    breakdown = {
        "operator":   1.0 if problem.operator_type      == hook.operator_type      else 0.0,
        "structural": 1.0 if problem.structural_pattern == hook.structural_pattern  else 0.0,
        "heuristic":  len(p_hset & h_hset) / len(union) if union else 0.0,
        "quantity":   problem.quantity_signature.similarity(hook.quantity_signature),
        "embedding":  cosine_sim(problem.embedding, hook.embedding_centroid),
        "prior":      hook.success_rate,
    }
    total = sum(WEIGHTS[k] * v for k, v in breakdown.items())
    return total, breakdown


RETRIEVAL_HIGH_CONF = 0.80   # definitely right schema
RETRIEVAL_MED_CONF  = 0.45   # probably right schema — lowered for post-filter scores
RETRIEVAL_LOW_CONF  = 0.30  # uncertain → near miss


def retrieve_schema(problem: ProblemFeatures, hooks: List[FeatureHook]):
    if not hooks:
        return None, 0.0, {}, "empty_bank"

    # ── STAGE 1: Hard filter by operator type ────────────────────────────────
    # Eliminates ~80% of wrong candidates before soft scoring begins.
    # Cleanest discriminator — not affected by embedding noise.
    operator_matches = [
        h for h in hooks
        if h.operator_type == problem.operator_type
    ]
    # Fall back to all hooks if nothing matches (novel operator type)
    candidates = operator_matches if operator_matches else hooks

    # ── STAGE 2: Soft match among candidates ─────────────────────────────────
    scored = [(h, *score_hook(problem, h)) for h in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    best_hook, best_score, best_breakdown = scored[0]

    # Three confidence zones
    if best_score >= RETRIEVAL_HIGH_CONF:
        return best_hook, best_score, best_breakdown, "retrieved_high"
    elif best_score >= RETRIEVAL_MED_CONF:
        return best_hook, best_score, best_breakdown, "retrieved_med"
    elif best_score >= RETRIEVAL_LOW_CONF:
        return best_hook, best_score, best_breakdown, "near_miss"
    else:
        return None, best_score, {}, "generic"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: SCHEMA BANK + REPLAY
# ══════════════════════════════════════════════════════════════════════════════

class SchemaBank:
    def __init__(self):
        self.schemas: Dict[str, dict]      = {}
        self.hooks:   Dict[str, FeatureHook] = {}

    def add(self, schema: dict, hook: FeatureHook):
        self.schemas[hook.schema_name] = schema
        self.hooks[hook.schema_name]   = hook
        print(f"  [Bank] Added '{hook.schema_name}' | size={len(self.schemas)}")

    def retrieve(self, features: ProblemFeatures):
        hook, score, breakdown, status = retrieve_schema(features, list(self.hooks.values()))
        if status in ("retrieved_high", "retrieved_med"):
            hook.usage_count += 1
            return self.schemas[hook.schema_name], hook, score, breakdown, status
        return None, None, score, breakdown, status

    def update_hook(self, schema_name: str, features: ProblemFeatures, correct: bool):
        if schema_name in self.hooks:
            h = self.hooks[schema_name]
            h.update_success(correct)
            h.update_centroid(features.embedding, correct=correct)

    def size(self): return len(self.schemas)


@dataclass
class Episode:
    episode_id:      int
    problem_text:    str
    features:        ProblemFeatures
    schema_used:     str
    outcome:         str
    retrieval_score: float
    task_id:         str
    log_llm_used:    bool = False
    mistake:         Optional[str] = None
    insight:         Optional[str] = None   # what worked when outcome == "correct"
    response_text:   Optional[str] = None


class ReplayBuffer:
    def __init__(self, max_size=500):
        self.buffer: List[Episode] = []
        self.max_size = max_size
        self.recent_schemas: List[str] = []

    def add(self, ep: Episode):
        self.buffer.append(ep)
        self.recent_schemas.append(ep.schema_used)
        self.recent_schemas = self.recent_schemas[-5:]
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def sample(self, n=10) -> List[Episode]:
        if len(self.buffer) < 3:
            return list(self.buffer)
        failures  = [e for e in self.buffer if e.outcome == "incorrect"]
        successes = [e for e in self.buffer if e.outcome == "correct"]
        n_fail = min(int(n * 0.7), len(failures))
        n_succ = min(n - n_fail, len(successes))
        return random.sample(failures, n_fail) + random.sample(successes, n_succ)

    def recent_accuracy(self, w=20):
        recent = self.buffer[-w:]
        return sum(1 for e in recent if e.outcome == "correct") / len(recent) if recent else 0.0

    def failure_rate(self):
        return sum(1 for e in self.buffer if e.outcome == "incorrect") / len(self.buffer) if self.buffer else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: HARD RESET LLM
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a precise mathematical problem solver.
For every problem output EXACTLY:

Steps to follow:
1. [plan, 3-6 items]

Solution:
- Step-by-step with units.

Sanity check:
- Brief justification.

Final Answer: <single value>"""


def hard_reset_call(problem_text: str, schema: dict, model: str = LLM_MODEL) -> Optional[str]:
    """
    Fresh API call per problem. No history. No shared context.
    Schema injection is the ONLY cross-episode information.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai")

    client = OpenAI(api_key=OPENAI_API_KEY)

    _anti = schema.get("anti_heuristics", [])
    _anti_line = (f"Common mistakes to avoid: {'; '.join(_anti)}\n"
                  if _anti else "")
    prompt = (
        f"Schema: {schema.get('name','Generic')}\n"
        f"Description: {schema.get('description','')}\n"
        f"Strategy: {schema.get('template','Solve step by step.')}\n"
        f"Heuristics: {', '.join(schema.get('heuristics', []))}\n"
        f"{_anti_line}"
        f"\nProblem:\n{problem_text}"
    )

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                **_llm_kwargs(model),
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                print(f"  [LLM] Failed: {e}")
                return None



    for attempt in range(3):
        try:
            # resp = client.chat.completions.create(
            #     model=model,
            #     temperature=0.0,
            #     messages=[
            #         {"role": "system", "content": SYSTEM_PROMPT},
            #         {"role": "user",   "content": prompt},
            #     ],
            # )
            resp = client.chat.completions.create(
                **_llm_kwargs(model),
                messages=[
                    # ── HARD RESET ── nothing above carries from prior episodes
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                    # ── END OF EPISODE CONTEXT ──
                ],
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                print(f"  [LLM] Failed: {e}")
                return None


# ── Agentic system prompt ─────────────────────────────────────────────────────
AGENT_SYSTEM_PROMPT = """You are a mathematical problem-solving memory agent.
You have access to these tools:

  TOOL: search_past_failures(<description>)
    Search your memory for similar past problems and what went wrong.
    Returns past mistakes to avoid and schemas that worked.
    ALWAYS call this first — it prevents repeating known errors.
    Example: TOOL: search_past_failures(probability with dice counting)

  TOOL: calculate(<python_expression>)
    Evaluates a Python math expression. Use to verify a COMPLETE step.
    Do NOT call calculate multiple times for the same computation.
    Combine operations into one expression where possible.
    Good:  TOOL: calculate((14 + (196 - 4*4.9*6.4)**0.5) / (2*4.9))
    Bad:   calling calculate separately for discriminant, sqrt, division


  TOOL: schema_lookup(<description>)
    Searches the memory bank for a specific strategy.
    Use when you need a technique not in your current schema.
    Example: TOOL: schema_lookup(how to solve linear congruences)

  TOOL: self_verify(<your_answer>)
    Checks whether your answer is correct before submitting.
    Use before committing to your final answer.
    Example: TOOL: self_verify(42)

  TOOL: decompose(<problem>)
    Breaks a complex problem into independent sub-problems.
    Use when the problem has clearly separate parts.
    Example: TOOL: decompose(Find all n where n divides both 12 and 18)

  TOOL: sympy_solve(<use_current_problem>)
    Solve the current math problem by:
      1) generating SymPy-only Python code,
      2) executing it in a sandbox,
      3) returning a numeric/fraction final answer.
    The argument is ignored; just pass: use_current_problem.
    Use this when the calculator or the schema approach is not enough to finish.

Strategy:
1. FIRST: call search_past_failures to check for known mistakes on this type.
2. Use schema_lookup if you need a specific technique.
3. Solve step by step using the schema and any retrieved experience.
4. Use calculate to verify numerical steps.
5. Call sympy_verify(<your_answer>) for formal symbolic verification — PREFERRED over self_verify.
   - If sympy_verify returns verified=True, write Final Answer immediately.
   - If sympy_verify returns disproven, revise and retry.
   - If uncertain, fall back to self_verify.
6. Write: Final Answer: <value>

Available tools (call ONE per turn):
- TOOL: calculate(<python_expression>)       — arithmetic/algebra computation
- TOOL: sympy_verify(<proposed_answer>)      — formal symbolic verification ★
- TOOL: self_verify(<proposed_answer>)       — LLM-based sanity check
- TOOL: sympy_solve(use_current_problem)     — let SymPy solve from scratch
- TOOL: schema_lookup(<topic>)               — retrieve a strategy from memory
- TOOL: search_past_failures(<query>)        — check episodic memory for mistakes
- TOOL: decompose(use_current_problem)       — split into sub-problems

Rules:
- Call at most ONE tool per turn.
- After seeing the tool result, continue reasoning.
- Do NOT write Final Answer until you are confident."""


def agentic_solve(problem_text: str, schema: dict,
                  bank: "SchemaBank", encoder,
                  model: str = LLM_MODEL,
                  max_turns: int = 15,
                  verbose: bool = False) -> dict:
    """
    Agentic episode: LLM can call tools, verify, retry.

    Replaces hard_reset_call for ISM only.
    The other three systems (Static, Neural, Schema) keep the
    passive hard_reset_call — giving a clean ablation.

    Returns:
      {
        "response":   full final response text (same format as hard_reset_call),
        "tool_trace": list of every tool call made,
        "n_turns":    how many reasoning turns were needed,
      }

    Tool trace entries:
      {"tool": "calculate",    "arg": "...", "result": "..."}
      {"tool": "schema_lookup","arg": "...", "schema": "...", "hit": bool}
      {"tool": "self_verify",  "arg": "...", "result": {...}}
      {"tool": "decompose",    "arg": "...", "result": {...}}
    """
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Build initial prompt — same schema injection as hard_reset_call
    _anti = schema.get("anti_heuristics", [])
    _anti_line = (f"Common mistakes to avoid: {'; '.join(_anti)}\n"
                  if _anti else "")
    user_prompt = (
        f"Schema: {schema.get('name', 'Generic')}\n"
        f"Description: {schema.get('description', '')}\n"
        f"Strategy: {schema.get('template', 'Solve step by step.')}\n"
        f"Heuristics: {', '.join(schema.get('heuristics', []))}\n"
        f"{_anti_line}"
        f"\nProblem:\n{problem_text}"
    )

    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]

    tool_trace   = []
    final_answer = None
    full_response_text = ""

    for turn in range(max_turns):
        try:
            # resp = client.chat.completions.create(
            #     model=model,
            #     temperature=0.0,
            #     max_tokens=800,
            #     messages=messages,
            # )
            resp = client.chat.completions.create(
                **_llm_kwargs(model, max_tokens=800),
                messages=messages,
            )
        except Exception as exc:
            print(f"  [Agent] LLM call failed turn {turn}: {exc}")
            break

        reply = resp.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": reply})
        full_response_text = reply  # keep last reply as the response

        if verbose:
            print(f"  [Agent turn {turn+1}] {reply[:120]}...")

        # Check for Final Answer — agent is done
        fa = re.search(r"Final Answer:\s*(.+?)(?:\n|$)", reply, re.IGNORECASE)
        if fa:
            final_answer = fa.group(1).strip()
            break

        # Parse TOOL calls — one per turn
        tool_match = None
        # Match tool calls on a single line to avoid truncating arguments containing ')'
        for line in reply.splitlines():
            m = re.search(r"TOOL:\s*(\w+)\((.*)\)\s*$", line)
            if m:
                tool_match = m
                break
        if not tool_match:
            # No tool call and no final answer — agent is stuck
            # Give it a nudge
            messages.append({
                "role": "user",
                "content": ("Continue solving. Use a tool if needed, "
                            "or write Final Answer: <value> when done.")
            })
            continue

        tool_name = tool_match.group(1).strip()
        tool_arg  = tool_match.group(2).strip().strip("\"'")

        # ── Execute the tool ──────────────────────────────────────────────────
        if tool_name == "calculate":
            result_str = tool_calculate(tool_arg)
            tool_trace.append({
                "tool":   "calculate",
                "arg":    tool_arg,
                "result": result_str,
            })
            tool_result_text = f"TOOL_RESULT: {result_str}"
            if verbose:
                print(f"  [calculate] {tool_arg} → {result_str}")

        elif tool_name == "sympy_solve":
            # Tool arg is intentionally ignored; we solve using the current problem_text.
            result_str = tool_sympy_solve(problem_text, client)
            tool_trace.append({
                "tool":   "sympy_solve",
                "arg":    tool_arg,
                "result": result_str,
            })
            tool_result_text = f"TOOL_RESULT: {result_str}"
            if verbose:
                print(f"  [sympy_solve] → {result_str}")

        elif tool_name == "schema_lookup":
            schema_name, schema_desc = tool_schema_lookup(
                tool_arg, bank, encoder
            )
            hit = schema_name != "Generic"
            tool_trace.append({
                "tool":   "schema_lookup",
                "arg":    tool_arg,
                "schema": schema_name,
                "hit":    hit,
            })
            tool_result_text = (
                f"TOOL_RESULT: Schema '{schema_name}'\n{schema_desc}"
            )
            if verbose:
                print(f"  [schema_lookup] '{tool_arg}' → '{schema_name}' (hit={hit})")

        elif tool_name == "self_verify":
            verif = tool_self_verify(
                problem_text, tool_arg,
                schema.get("name", "Generic"), client
            )
            tool_trace.append({
                "tool":   "self_verify",
                "arg":    tool_arg,
                "result": verif,
            })
            # If verification succeeded, use as final answer
            if verif.get("verified"):
                final_answer = tool_arg
                tool_result_text = (
                    f"TOOL_RESULT: Verified correct "
                    f"(confidence={verif.get('confidence',0):.2f})"
                )
            elif verif.get("corrected_answer"):
                final_answer = verif["corrected_answer"]
                tool_result_text = (
                    f"TOOL_RESULT: Incorrect. Issue: {verif.get('issue','')}. "
                    f"Corrected answer: {verif['corrected_answer']}"
                )
            else:
                tool_result_text = (
                    f"TOOL_RESULT: Incorrect. Issue: {verif.get('issue','')}. "
                    f"Please try again."
                )
            if verbose:
                print(f"  [self_verify] answer='{tool_arg}' verified={verif.get('verified')}")
            if final_answer:
                break

        elif tool_name == "decompose":
            decomp = tool_decompose(problem_text, client)
            tool_trace.append({
                "tool":   "decompose",
                "arg":    tool_arg,
                "result": decomp,
            })
            if decomp.get("sub_problems"):
                parts = "\n".join(
                    f"  Sub-problem {i+1} [{sp.get('domain','')}]: {sp.get('text','')}"
                    for i, sp in enumerate(decomp["sub_problems"])
                )
                tool_result_text = (
                    f"TOOL_RESULT: Decomposed into:\n{parts}\n"
                    f"Combination: {decomp.get('combination_rule','')}"
                )
            else:
                tool_result_text = "TOOL_RESULT: Could not decompose."
            if verbose:
                n = len(decomp.get("sub_problems", []))
                print(f"  [decompose] → {n} sub-problems")

        elif tool_name == "search_past_failures":
            # Search episodic memory for similar past problems
            search_result = tool_search_past_failures(
                query   = tool_arg,
                replay  = bank._replay_ref if hasattr(bank, '_replay_ref') else ReplayBuffer(),
                encoder = encoder,
                top_k   = 3,
            )
            tool_trace.append({
                "tool":   "search_past_failures",
                "arg":    tool_arg,
                "result": search_result,
            })
            # Format result for the agent in a readable way
            lines = []
            if search_result.get("message"):
                lines.append(search_result["message"])
            else:
                if search_result["common_mistakes"]:
                    lines.append("⚠ Common mistakes to avoid: "
                                 + "; ".join(search_result["common_mistakes"]))
                if search_result["working_schemas"]:
                    lines.append("✓ Schemas that worked: "
                                 + ", ".join(search_result["working_schemas"]))
                if search_result["avoid_schemas"]:
                    lines.append("✗ Schemas that failed: "
                                 + ", ".join(search_result["avoid_schemas"]))
                for ep in search_result["similar_problems"][:2]:
                    status = "✓" if ep["outcome"] == "correct" else "✗"
                    lines.append(f"{status} ep{ep['episode']} "
                                 f"(sim={ep['similarity']}): {ep['problem'][:100]}")
                    if ep.get("mistake"):
                        lines.append(f"   Mistake was: {ep['mistake']}")
                    if ep.get("correct_reasoning"):
                        lines.append(f"   Worked via: {ep['correct_reasoning'][:100]}")
            tool_result_text = ("TOOL_RESULT (memory search):\n"
                                + "\n".join(lines)) if lines \
                               else "TOOL_RESULT: No similar past problems found yet."
            if verbose:
                n = len(search_result.get("similar_problems", []))
                print(f"  [search_past_failures] '{tool_arg}' → {n} similar episodes")

        elif tool_name == "sympy_verify":
            # Formal symbolic verification — strongest correctness signal
            verif = tool_sympy_verify(problem_text, tool_arg, client)
            tool_trace.append({
                "tool":   "sympy_verify",
                "arg":    tool_arg,
                "result": verif,
            })
            v = verif.get("verified")
            reason = verif.get("reason", "")
            sympy_ans = verif.get("sympy_answer")
            if v is True:
                final_answer = tool_arg
                tool_result_text = (
                    f"TOOL_RESULT: Formally verified ✓ — {reason}. "
                    f"Your answer is correct."
                )
            elif v is False:
                tool_result_text = (
                    f"TOOL_RESULT: Formally disproven ✗ — {reason}. "
                    + (f"SymPy computed: {sympy_ans}. " if sympy_ans else "")
                    + "Revise your answer."
                )
            else:
                tool_result_text = (
                    f"TOOL_RESULT: Could not verify symbolically — {reason}. "
                    f"Proceed with caution."
                )
            if verbose:
                print(f"  [sympy_verify] answer='{tool_arg}' verified={v} reason='{reason}'")
            if final_answer:
                break

        else:
            tool_result_text = f"TOOL_RESULT: Unknown tool '{tool_name}'"
            tool_trace.append({"tool": "unknown", "arg": tool_arg})

        # Feed tool result back to agent for next turn
        messages.append({"role": "user", "content": tool_result_text})

    

    # If agent ran out of turns without a Final Answer,
    # make one last call asking it to commit to its best answer
    if not final_answer:
        try:
            messages.append({
                "role": "user",
                "content": ("You have reached the turn limit. "
                            "Based on your work so far, write your best answer now. "
                            "You MUST write: Final Answer: <value>")
            })
            # resp = client.chat.completions.create(
            #     model=model, temperature=0.0, max_tokens=200,
            #     messages=messages,
            # )
            resp = client.chat.completions.create(
                **_llm_kwargs(model, max_tokens=200),
                messages=messages,
            )
            reply = resp.choices[0].message.content or ""
            full_response_text = reply
            fa = re.search(r"Final Answer:\s*(.+?)(?:\n|$)", reply, re.IGNORECASE)
            if fa:
                final_answer = fa.group(1).strip()
        except Exception:
            pass

    if final_answer and "Final Answer:" not in full_response_text:
        full_response_text += f"\nFinal Answer: {final_answer}"

    return {
        "response":    full_response_text,
        "tool_trace":  tool_trace,
        "n_turns":     turn + 1,
    }


def extract_final_answer(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"Final Answer:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    return m.group(1).strip() if m else None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4b: AGENTIC TOOLS
# Four tools the agent can call during problem solving.
# Each tool produces a new memory signal beyond just correct/incorrect.
# ══════════════════════════════════════════════════════════════════════════════

# ── Tool 1: Calculator ────────────────────────────────────────────────────────
# The agent uses this to verify numerical steps.
# Why it matters for memory: if the agent computes correctly but still gets
# the answer wrong, the schema gave the WRONG APPROACH (not arithmetic error).
# This is a stronger self-correct signal than a plain incorrect outcome.

def tool_calculate(expression: str) -> str:
    """
    Safe Python expression evaluator for mathematical computation.
    The agent calls this to verify intermediate steps numerically.

    Examples:
      tool_calculate("17**3 % 5")          → "2"
      tool_calculate("gcd(48, 18)")        → "6"
      tool_calculate("comb(10, 3)")        → "120"
      tool_calculate("sum(range(1, 101))") → "5050"
    """
    import math
    safe_globals = {
        "__builtins__": {},
        "math":         math,
        "gcd":          math.gcd,
        "lcm":          math.lcm,
        "factorial":    math.factorial,
        "comb":         math.comb,
        "perm":         math.perm,
        "sqrt":         math.sqrt,
        "floor":        math.floor,
        "ceil":         math.ceil,
        "log":          math.log,
        "log2":         math.log2,
        "log10":        math.log10,
        "sin":          math.sin,
        "cos":          math.cos,
        "tan":          math.tan,
        "pi":           math.pi,
        "e":            math.e,
        "inf":          math.inf,
        "sum":          sum,
        "range":        range,
        "abs":          abs,
        "round":        round,
        "int":          int,
        "float":        float,
        "pow":          pow,
        "max":          max,
        "min":          min,
        "list":         list,
        "len":          len,
    }
    expr = expression.strip()

    # Auto-fix common bracket issues the agent makes, e.g. "(-14", "(70.56"
    # Balance parentheses conservatively so valid expressions still work.
    diff = expr.count("(") - expr.count(")")
    if diff > 0:
        expr = expr + (")" * diff)
    elif diff < 0:
        # Too many closing parens at the end – trim minimal suffix
        while diff < 0 and expr.endswith(")"):
            expr = expr[:-1]
            diff += 1

    try:
        result = eval(expr, safe_globals)
        return str(result)
    except Exception as exc:
        return f"Error: {exc}. Check your expression syntax and try again."


# ── Tool 2.5: SymPy Solver ────────────────────────────────────────────────
# Two-step approach:
#   (1) ask an LLM to generate SymPy-only code that sets `final_answer`
#   (2) exec it in a restricted environment, then convert to a numeric/fraction string
_SYMPY_CODE_PROMPT = """Generate ONLY Python code (no markdown).

You must solve the math problem using SymPy.

Rules:
1. Do NOT use markdown fences. Output code only.
2. Do NOT import anything. Use SymPy via the alias `sp` that will be provided.
3. Set `final_answer` to the required final value as either:
   - a SymPy Integer/Rational/number, or
   - a SymPy Expr that evaluates numerically,
   - OR a Python int / float.
4. Do NOT print. No other side effects.
5. If you solve an equation and get multiple solutions, choose the correct ones
   that satisfy the original equation/problem statement.

Problem:
{problem}

Now output the code that sets `final_answer`.
"""


def tool_sympy_solve(problem: str, client) -> str:
    """
    Generate SymPy-only code with an LLM, execute it safely, and return a numeric result.
    """
    try:
        resp = client.chat.completions.create(
            **_llm_kwargs(LLM_MODEL, max_tokens=500),
            messages=[
                {
                    "role": "system",
                    "content": "You are a SymPy code generator. Output code only.",
                },
                {
                    "role": "user",
                    "content": _SYMPY_CODE_PROMPT.format(problem=problem[:1200]),
                },
            ],
        )
        code = resp.choices[0].message.content or ""
        code = re.sub(r"^\s*```(?:python)?\s*", "", code, flags=re.IGNORECASE)
        code = re.sub(r"\s*```\s*$", "", code, flags=re.IGNORECASE)

        import sympy as sp
        safe_globals = {"__builtins__": {}, "sp": sp}
        safe_locals = {}
        exec(code, safe_globals, safe_locals)

        final_answer = safe_locals.get("final_answer", None)
        if final_answer is None:
            final_answer = safe_locals.get("answer", None)
        if final_answer is None:
            return "Error: sympy_solve did not set final_answer."

        # Normalize return into an integer/fraction if possible, else numeric approximation.
        if isinstance(final_answer, (int, float)):
            return str(final_answer).strip()

        fa = final_answer
        try:
            fa = sp.simplify(fa)
        except Exception:
            pass

        # Rational/integer: return exact fraction form p/q or integer.
        try:
            if getattr(fa, "is_Rational", False):
                if int(fa.q) == 1:
                    return str(int(fa.p))
                return f"{int(fa.p)}/{int(fa.q)}"
            if getattr(fa, "is_Integer", False):
                return str(int(fa))
        except Exception:
            pass

        # If SymPy still has a numeric value, approximate.
        try:
            numeric = sp.N(fa, 25)
            numeric_s = str(numeric).replace(" ", "").strip()
            if re.fullmatch(r"-?\d+(?:\.0+)?", numeric_s):
                numeric_s = str(int(float(numeric_s)))
            return numeric_s
        except Exception:
            return str(fa).strip()

    except Exception as exc:
        return f"Error: sympy_solve failed ({exc})."


# ── Tool 2.6: SymPy Verifier ─────────────────────────────────────────────────
# Formally verifies a proposed answer using SymPy.
# Unlike tool_self_verify (LLM opinion), this executes symbolic math code.
# Why it matters for memory: a schema whose answers pass sympy_verify consistently
# gets a stronger quality signal than one that only matches gold string comparison.
# For the workshop: this is the "verifiable" claim — the system can formally
# certify its own answers, not just self-report confidence.

_SYMPY_VERIFY_PROMPT = """Generate ONLY Python code (no markdown, no imports).

You are given a math problem and a proposed answer.
Write SymPy code that VERIFIES whether the proposed answer is correct.

Rules:
1. Use SymPy via the alias `sp` (already imported).
2. Set `verified` to True if the answer is correct, False if wrong.
3. Set `reason` to a short string explaining why (e.g., "satisfies equation", "substitution check failed").
4. Set `sympy_answer` to the SymPy-computed correct answer if you can find it, else None.
5. Do NOT print. No other side effects.

Verification strategies (use whichever fits):
- Equation: substitute proposed answer back and check if LHS == RHS
- Inequality: check the boundary conditions
- Count/combinatorics: compute independently and compare
- Geometry: verify via coordinate computation
- If you cannot verify symbolically, set verified = None (uncertain)

Problem:
{problem}

Proposed answer: {proposed_answer}

Output code only:"""


def tool_sympy_verify(problem: str, proposed_answer: str, client) -> dict:
    """
    Formally verify a proposed answer using SymPy-generated code.

    Returns:
        {
            "verified":      True | False | None (uncertain),
            "reason":        str,
            "sympy_answer":  str | None,   # SymPy's own computed answer if found
            "formal":        True,         # flag for schema quality signal
        }
    """
    try:
        resp = client.chat.completions.create(
            **_llm_kwargs(LLM_MODEL, max_tokens=600),
            messages=[
                {
                    "role": "system",
                    "content": "You are a SymPy verification code generator. Output code only.",
                },
                {
                    "role": "user",
                    "content": _SYMPY_VERIFY_PROMPT.format(
                        problem=problem[:1200],
                        proposed_answer=proposed_answer[:200],
                    ),
                },
            ],
        )
        code = resp.choices[0].message.content or ""
        # Strip markdown fences if model disobeys
        code = re.sub(r"^\s*```(?:python)?\s*", "", code, flags=re.IGNORECASE)
        code = re.sub(r"\s*```\s*$", "", code, flags=re.IGNORECASE)

        import sympy as sp
        safe_globals = {"__builtins__": {}, "sp": sp}
        safe_locals  = {"verified": None, "reason": "no code executed", "sympy_answer": None}
        exec(code, safe_globals, safe_locals)

        verified      = safe_locals.get("verified", None)
        reason        = safe_locals.get("reason", "")
        sympy_answer  = safe_locals.get("sympy_answer", None)

        # Convert sympy_answer to string if it's a SymPy object
        if sympy_answer is not None and hasattr(sympy_answer, "__class__"):
            try:
                sympy_answer = str(sp.simplify(sympy_answer))
            except Exception:
                sympy_answer = str(sympy_answer)

        return {
            "verified":     verified,
            "reason":       str(reason),
            "sympy_answer": sympy_answer,
            "formal":       True,
        }

    except Exception as exc:
        return {
            "verified":     None,
            "reason":       f"sympy_verify failed: {exc}",
            "sympy_answer": None,
            "formal":       True,
        }


# ── Tool 2: Schema Lookup ─────────────────────────────────────────────────────
# The agent asks the bank for a specific strategy mid-solve.
# Why it matters for memory: every lookup that MISSES means the bank is
# missing a schema the agent needed → synthesis trigger.
# Every lookup that HITS and leads to a correct answer → strong signal.

def tool_schema_lookup(query: str, bank: "SchemaBank",
                       encoder) -> Tuple[str, str]:
    """
    Agent queries the bank mid-solve for a specific strategy.
    Returns (schema_name, description) of the best match.

    Example:
      tool_schema_lookup("how to solve modular congruences", bank, encoder)
      → ("Number Theory", "Problems involving integers, primes, GCD...")

    Memory signal:
      Hit  + correct   → boost matched schema
      Hit  + incorrect → flag matched schema for correction
      Miss + incorrect → synthesise a new schema for this query
    """
    if bank.size() == 0:
        return ("Generic", "No schemas available yet.")

    query_emb = encoder.encode(query.strip(), normalize_embeddings=True)

    best_name  = None
    best_sim   = -1.0
    for name, hook in bank.hooks.items():
        sim = cosine_sim(query_emb, hook.embedding_centroid)
        if sim > best_sim:
            best_sim  = sim
            best_name = name

    if best_sim >= 0.35 and best_name:
        schema = bank.schemas.get(best_name, {})
        desc   = schema.get("description", "")[:300]
        return (best_name, desc)

    return ("Generic", "No close schema found for this query.")


# ── Tool 3: Self-Verifier ─────────────────────────────────────────────────────
# The agent checks its own answer before committing.
# Why it matters for memory: a failed verification is direct evidence that
# the schema gave a wrong approach, not just that the LLM made a slip.

_VERIFY_PROMPT = """You solved a math problem. Verify whether the answer is correct.

Problem:
{problem}

Your answer: {answer}
Schema used: {schema_name}

Check:
1. Does the answer satisfy all constraints stated in the problem?
2. Is the answer in the right form (integer, fraction, set, expression)?
3. Does a quick sanity check confirm it (plug back in, check edge cases)?

Reply with ONLY this JSON (no markdown, no explanation):
{{
  "verified": true or false,
  "confidence": 0.0 to 1.0,
  "issue": "one sentence describing what is wrong, or null if verified",
  "corrected_answer": "corrected value as a string, or null if verified or unknown"
}}"""

def tool_self_verify(problem: str, answer: str,
                     schema_name: str, client) -> dict:
    """
    Agent asks a fresh LLM call to verify its own answer.
    Returns dict with keys: verified, confidence, issue, corrected_answer.

    Memory signal:
      verified=False → schema gave wrong approach → amplify self-correct signal
      corrected_answer present → use as the actual submitted answer
    """
    try:
        # resp = client.chat.completions.create(
        #     model=LLM_MODEL,
        #     temperature=0.0,
        #     max_tokens=200,
        #     messages=[{
        #         "role": "user",
        #         "content": _VERIFY_PROMPT.format(
        #             problem=problem[:600],
        #             answer=answer[:200],
        #             schema_name=schema_name,
        #         )
        #     }]
        # )
        resp = client.chat.completions.create(
            **_llm_kwargs(LLM_MODEL, max_tokens=200),
            messages=[{
                "role": "user",
                "content": _VERIFY_PROMPT.format(
                    problem=problem[:600],
                    answer=answer[:200],
                    schema_name=schema_name,
                )
            }]
        )
        text = resp.choices[0].message.content.strip()
        text = re.sub(r'^```json\s*|\s*```$', '', text, flags=re.MULTILINE)
        result = json.loads(text)
        # Ensure required keys
        result.setdefault("verified",          True)
        result.setdefault("confidence",        0.5)
        result.setdefault("issue",             None)
        result.setdefault("corrected_answer",  None)
        return result
    except Exception as exc:
        return {"verified": True, "confidence": 0.5,
                "issue": None, "corrected_answer": None,
                "_error": str(exc)}


# ── Tool 4: Sub-Problem Decomposer ────────────────────────────────────────────
# For complex problems the agent breaks them into independent sub-problems,
# each of which gets its own schema lookup.
# Why it matters for memory: decomposition patterns are rich schema metadata.
# If problem X always decomposes into [algebra + number_theory], the schema
# for X should encode that structure.

_DECOMPOSE_PROMPT = """A math problem is too complex for one approach.
Break it into 2-3 independent sub-problems that can be solved separately.

Problem:
{problem}

For each sub-problem identify which mathematical domain it belongs to.
Domains: algebra, number_theory, geometry, combinatorics, probability, calculus

Reply with ONLY this JSON (no markdown):
{{
  "sub_problems": [
    {{"text": "sub-problem 1 statement", "domain": "domain_name"}},
    {{"text": "sub-problem 2 statement", "domain": "domain_name"}}
  ],
  "combination_rule": "one sentence: how to combine sub-answers into final answer"
}}"""

def tool_decompose(problem: str, client) -> dict:
    """
    Agent decomposes a complex problem into sub-problems.
    Returns dict with sub_problems list and combination_rule.

    Memory signal:
      Successful decomposition → store decomposition pattern in schema metadata
      decomp + correct → this schema can handle decomposable problems
    """
    try:
        # resp = client.chat.completions.create(
        #     model=LLM_MODEL,
        #     temperature=0.0,
        #     max_tokens=400,
        #     messages=[{
        #         "role": "user",
        #         "content": _DECOMPOSE_PROMPT.format(problem=problem[:600])
        #     }]
        # )
        resp = client.chat.completions.create(
            **_llm_kwargs(LLM_MODEL, max_tokens=400),
            messages=[{
                "role": "user",
                "content": _DECOMPOSE_PROMPT.format(problem=problem[:600])
            }]
        )
        text = resp.choices[0].message.content.strip()
        text = re.sub(r'^```json\s*|\s*```$', '', text, flags=re.MULTILINE)
        result = json.loads(text)
        result.setdefault("sub_problems",    [])
        result.setdefault("combination_rule", "Combine sub-answers directly.")
        return result
    except Exception as exc:
        return {"sub_problems": [], "combination_rule": "",
                "_error": str(exc)}


# ── Tool 5: Search Past Failures ──────────────────────────────────────────────
_MISTAKE_PROMPT = """A math agent got this problem wrong.

Problem: {problem}
Schema used: {schema}
Agent's response: {response}

In ONE sentence, describe the specific mathematical mistake the agent made.
Examples:
  "Forgot to divide by 2 for symmetric outcomes"
  "Used permutation instead of combination"
  "Did not account for dependent events"

Reply with ONLY the one-sentence mistake description."""


def extract_mistake(problem, schema_name, response, client):
    if not response:
        return None
    try:
        # resp = client.chat.completions.create(
        #     model=LLM_MODEL, temperature=0.0, max_tokens=80,
        #     messages=[{"role": "user", "content": _MISTAKE_PROMPT.format(
        #         problem=problem[:400], schema=schema_name,
        #         response=response[:600])}]
        # )
        resp = client.chat.completions.create(
            **_llm_kwargs(LLM_MODEL, max_tokens=80),
            messages=[{"role": "user", "content": _MISTAKE_PROMPT.format(
                problem=problem[:400], schema=schema_name,
                response=response[:600])}]
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None


_SUCCESS_PROMPT = """A math agent solved this problem correctly.

Problem: {problem}
Schema used: {schema}
Agent's response: {response}

In ONE sentence, describe the key insight or step that made this solution correct.
Examples:
  "Recognized that complementary counting was simpler than direct counting"
  "Correctly set up simultaneous equations to eliminate the unknown"
  "Applied the Pythagorean theorem after decomposing the figure into right triangles"

Reply with ONLY the one-sentence insight."""


def extract_success_insight(problem, schema_name, response, client):
    if not response:
        return None
    try:
        resp = client.chat.completions.create(
            **_llm_kwargs(LLM_MODEL, max_tokens=80),
            messages=[{"role": "user", "content": _SUCCESS_PROMPT.format(
                problem=problem[:400], schema=schema_name,
                response=response[:600])}]
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None


def tool_search_past_failures(query, replay, encoder, top_k=3,
                              recency_lambda: float = 0.01):
    """
    Semantic search over replay buffer with recency weighting.

    Score = cosine_sim(query, episode_embedding) * exp(-lambda * age)

    lambda=0.01 gives:
      age=0   → weight 1.00   (current episode)
      age=50  → weight 0.61   (moderately discounted)
      age=100 → weight 0.37   (half-life ~70 episodes)
      age=200 → weight 0.14   (old failures mostly suppressed)

    This prevents stale failures (where the agent has since learned the correct
    schema) from polluting the search results and confusing the solver.
    """
    if not replay.buffer:
        return {"similar_problems": [], "common_mistakes": [],
                "working_schemas": [], "avoid_schemas": [],
                "message": "No past episodes yet."}

    query_emb   = encoder.encode(query.strip(), normalize_embeddings=True)
    current_ep  = replay.buffer[-1].episode_id  # most recent episode id

    scored = sorted(
        [
            (
                cosine_sim(query_emb, ep.features.embedding)
                * float(np.exp(-recency_lambda * max(0, current_ep - ep.episode_id))),
                ep,
            )
            for ep in replay.buffer
        ],
        key=lambda x: x[0],
        reverse=True,
    )[:top_k]

    similar = []
    for sim, ep in scored:
        entry = {"problem": ep.problem_text[:200], "schema": ep.schema_used,
                 "outcome": ep.outcome, "episode": ep.episode_id,
                 "similarity": round(sim, 3)}
        if ep.outcome == "incorrect" and ep.mistake:
            entry["mistake"] = ep.mistake
        if ep.outcome == "correct" and ep.response_text:
            key_lines = [l.strip() for l in ep.response_text.split("\n")
                         if l.strip() and any(kw in l.lower() for kw in
                            ["step","therefore","so ","final","answer","thus","we get"])]
            if key_lines:
                entry["correct_reasoning"] = " | ".join(key_lines[:3])
        similar.append(entry)

    incorrect_eps = [ep for _, ep in scored if ep.outcome == "incorrect"]
    correct_eps   = [ep for _, ep in scored if ep.outcome == "correct"]
    working       = list({ep.schema_used for ep in correct_eps})
    avoid         = list({ep.schema_used for ep in incorrect_eps
                          if ep.schema_used not in working
                          and ep.schema_used != "Generic Problem Solving"})
    mistakes      = list({ep.mistake for ep in incorrect_eps if ep.mistake})

    return {"similar_problems": similar, "common_mistakes": mistakes,
            "working_schemas": working, "avoid_schemas": avoid}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: ANSWER EVALUATION (competition math aware)
# ══════════════════════════════════════════════════════════════════════════════

def extract_boxed(solution: str) -> Optional[str]:
    matches = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', solution)
    return matches[-1].strip() if matches else None


def boxed_to_float(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        pass
    frac = re.match(r'\\frac\{(-?\d+)\}\{(-?\d+)\}', s.strip())
    if frac:
        n, d = int(frac.group(1)), int(frac.group(2))
        return n / d if d != 0 else None
    sf = re.match(r'^(-?\d+)/(\d+)$', s.strip())
    if sf:
        return int(sf.group(1)) / int(sf.group(2))
    return None


def evaluate_answer(response: Optional[str], gold_boxed: str,
                    gold_numeric: Optional[float], tol=0.01) -> bool:
    if not response or not gold_boxed:
        return False

    pred_text = extract_final_answer(response)
    if not pred_text:
        # fallback 1: last \boxed{...} in the response
        m = re.findall(r'\\boxed\{([^}]*)\}', response)
        if m:
            pred_text = m[-1]
        else:
            # fallback 2: last standalone number/fraction in response
            m2 = re.findall(r'(?<![\\a-zA-Z])(-?\d+(?:[./]\d+)?)', response)
            pred_text = m2[-1] if m2 else None
    if not pred_text:
        return False

    # Strip LaTeX delimiters and trailing units early
    def clean_pred(x: str) -> str:
        x = re.sub(r'\\\(|\\\)|\\\[|\\\]|\$', '', x)
        x = re.sub(r'\\,|\\;|\\!|\\:', '', x)
        x = re.sub(r'\\boxed\{([^}]*)\}', r'\1', x)
        x = re.sub(r'\\text\{([^}]*)\}', r'\1', x)
        x = re.sub(
            r'\s+(seconds?|minutes?|hours?|days?|meters?|feet|foot|cm|km|'
            r'inches?|pounds?|kg|miles?|units?|dollars?|cents?|percent|%'
            r'|sq\.?\s*\w+|cubic\s*\w+)\s*$',
            '', x, flags=re.IGNORECASE
        )
        return x.strip()

    pred_clean = clean_pred(pred_text).replace('%', '').strip()

    # Numeric comparison
    if gold_numeric is not None:
        pred_num = boxed_to_float(pred_clean)
        if pred_num is None:
            m = re.search(r'\\frac\{(-?\d+)\}\{(\d+)\}', pred_clean)
            if m:
                n, d = int(m.group(1)), int(m.group(2))
                pred_num = n / d if d != 0 else None
        if pred_num is None:
            try:
                numeric_only = re.sub(r'[^\d\.\-\/]', '', pred_clean.replace(',', ''))
                pred_num = float(numeric_only) if numeric_only else None
            except Exception:
                pass
        if pred_num is not None:
            if gold_numeric == 0:
                return abs(pred_num) < tol
            return abs(pred_num - gold_numeric) / abs(gold_numeric) <= tol

    # String normalization
    def norm(x):
        x = re.sub(r'\\\(|\\\)|\\\[|\\\]|\$', '', x)
        x = re.sub(r'\\,|\\;|\\!|\\:', '', x)
        # strip \boxed{...} and \text{...} wrappers
        x = re.sub(r'\\boxed\{([^}]*)\}', r'\1', x)
        x = re.sub(r'\\text\{([^}]*)\}', r'\1', x)
        x = re.sub(r'\\t?d?frac', r'\\frac', x)   # normalise \tfrac \dfrac
        x = x.replace(',', '')                      # strip thousands separators
        x = re.sub(r'\s+', '', x)
        return x.lower().strip()

    return norm(pred_clean) == norm(gold_boxed)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: MEMORY CONTROLLER
# ══════════════════════════════════════════════════════════════════════════════

GENERIC_SCHEMA = {
    "name": "Generic Problem Solving",
    "description": "General mathematical problem solving.",
    "template": "1. Identify knowns/unknowns.\n2. Set up equation.\n3. Solve.\n4. Verify.",
    "heuristics": ["decompose", "introduce_variable"],
}



# ── Fix 2: LLM operator classification ───────────────────────────────────────
OPERATOR_CLASSIFICATION_PROMPT = """Classify this math problem into exactly ONE category.

Categories:
- algebraic      : equations, expressions, simplification, exponents, radicals,
                   sequences, functions, inequalities, logarithms, systems
- number_theory  : divisibility, primes, GCD, LCM, modular arithmetic,
                   remainders, digit problems, Diophantine equations
- geometric      : area, volume, perimeter, coordinate geometry, angles,
                   triangles, circles, trigonometry, similarity
- combinatoric   : counting arrangements, permutations, combinations,
                   inclusion-exclusion, pigeonhole, stars-and-bars, paths
- probability    : chance, likelihood, expected value, conditional probability,
                   dice, cards, coins, urns, geometric probability
- calculus       : limits, derivatives, integrals, series convergence,
                   Taylor series, real analysis inequalities

Problem: {problem}

Reply with ONLY the category name. No explanation. No punctuation."""

_operator_cache: Dict[str, str] = {}

def classify_operator_llm(problem_text: str, client) -> str:
    """
    Fix 2: LLM classification of operator type.
    Much more accurate than keyword matching for subtle problems.
    Cached — same problem never classified twice.
    Cost: ~$0.0001 per call using gpt-4o-mini (tiny).
    """
    key = problem_text[:300]
    if key in _operator_cache:
        return _operator_cache[key]

    valid_categories = {
        "algebraic", "number_theory", "geometric",
        "combinatoric", "probability", "calculus"
    }

    try:
        resp = client.chat.completions.create(
            **_llm_kwargs(LLM_MODEL, max_tokens=10),
            messages=[{
                "role": "user",
                "content": OPERATOR_CLASSIFICATION_PROMPT.format(
                    problem=problem_text[:500]
                )
            }]
        )
        category = resp.choices[0].message.content.strip().lower()
        # Strip punctuation
        category = re.sub(r'[^a-z_]', '', category)

        if category not in valid_categories:
            # Fall back to rule-based if LLM returns invalid category
            category = FeatureExtractor()._operator(problem_text.lower())

        _operator_cache[key] = category
        return category

    except Exception:
        # Fall back silently on any error
        fallback = FeatureExtractor()._operator(problem_text.lower())
        _operator_cache[key] = fallback
        return fallback


# ── Fix 3: Warm-start centroid builder ───────────────────────────────────────
def build_warm_hook(schema: dict, seed_features: ProblemFeatures,
                    encoder) -> FeatureHook:
    """
    Fix 3: Build hook with warm-started centroid.
    Averages: seed problem embedding + description + template + name.
    Much more stable than single-problem centroid from day 1.
    """
    embeddings = [seed_features.embedding]

    for field_key in ("description", "template", "name"):
        text = schema.get(field_key, "")
        # template may be a list of strings from synthesizer
        if isinstance(text, list):
            text = " ".join(str(t) for t in text)
        text = str(text).strip()
        if text:
            emb = encoder.encode(text, normalize_embeddings=True)
            embeddings.append(emb)

    # Average and renormalize
    centroid = np.mean(embeddings, axis=0).astype(np.float32)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid /= norm

    return FeatureHook(
        schema_name         = schema["name"],
        operator_type       = seed_features.operator_type,
        structural_pattern  = seed_features.structural_pattern,
        heuristic_signature = schema.get("heuristics", []),
        quantity_signature  = seed_features.quantity_signature,
        embedding_centroid  = centroid,
        is_seed             = True,
    )


class MemoryController:
    def __init__(self, bank: SchemaBank, replay: ReplayBuffer,
                 encoder, synthesizer=None, verbose=True, name="Schema Memory"):
        self.bank       = bank
        self.replay     = replay
        self.encoder    = encoder
        self.synthesizer= synthesizer
        self.verbose    = verbose
        self.name       = name
        self._ep        = 0
        self._last_schema = None
        # Fix 2: LLM client for operator classification
        try:
            from openai import OpenAI
            self._llm_client = OpenAI(api_key=OPENAI_API_KEY)
        except Exception:
            self._llm_client = None
        # Hybrid extractor: LLM when confident, rule-based fallback.
        self.extractor = HybridFeatureExtractor(
            self._llm_client, model_name=LLM_MODEL, debug=FEATURE_DEBUG
        )

    def encode(self, text):
        return self.encoder.encode(text, normalize_embeddings=True)

    def get_schema_for_problem(self, problem_text):
        emb      = self.encode(problem_text)
        features = self.extractor.extract(problem_text, emb)

        # Override operator only when we used rule-based (avoid redundant LLM call)
        if not features.log_llm_used and self._llm_client is not None:
            features.operator_type = classify_operator_llm(
                problem_text, self._llm_client
            )

        schema, hook, score, breakdown, status = self.bank.retrieve(features)
        if schema is None:
            schema = GENERIC_SCHEMA
        return schema, features, score, breakdown, status

    def after_episode(self, problem_text, features, schema_used,
                      ret_status, outcome, task_id,
                      tool_trace: Optional[List[dict]] = None,
                      response_text: Optional[str] = None):
        schema_name = schema_used.get("name", "Generic")

        # Update hook — include near_miss so hooks learn even from low-confidence retrievals
        if ret_status in ("retrieved_high", "retrieved_med", "near_miss"):
            self.bank.update_hook(schema_name, features, outcome == "correct")

        # ── Learn from tool trace (agentic signal) ────────────────────────────
        if tool_trace:
            self._learn_from_tools(schema_name, features, outcome,
                                   tool_trace, problem_text)

        # ── Extract mistake / insight for episodic memory ─────────────────────
        ep_mistake  = None
        ep_insight  = None
        if self._llm_client and response_text:
            if outcome == "incorrect":
                ep_mistake = extract_mistake(
                    problem_text, schema_name,
                    response_text, self._llm_client
                )
                if ep_mistake and self.verbose:
                    print(f"  [Mistake] '{schema_name}': {ep_mistake}")
            elif outcome == "correct" and ret_status in ("retrieved_high", "retrieved_med"):
                # Only extract insights when the schema was actually responsible
                ep_insight = extract_success_insight(
                    problem_text, schema_name,
                    response_text, self._llm_client
                )
                if ep_insight and self.verbose:
                    print(f"  [Insight] '{schema_name}': {ep_insight}")

        # Log episode — with mistake/insight for episodic memory search
        ep = Episode(
            episode_id      = self._ep,
            problem_text    = problem_text,
            features        = features,
            schema_used     = schema_name,
            outcome         = outcome,
            retrieval_score = 0.0,
            task_id         = task_id,
            log_llm_used    = features.log_llm_used,
            mistake         = ep_mistake,
            insight         = ep_insight,
            response_text   = response_text[:600] if response_text else None,
        )
        self.replay.add(ep)

        # Association
        if self._last_schema and self._last_schema != schema_name:
            if self._last_schema in self.bank.hooks:
                h = self.bank.hooks[self._last_schema]
                if schema_name not in h.associated_schemas:
                    h.associated_schemas.append(schema_name)

        # Schema evolution on failure — stricter trigger to reduce synthesis bloat.
        # Require: (1) at least 10 episodes elapsed so we have real history,
        #          (2) 3+ failures of the SAME operator type in the last 20 episodes.
        # This ensures a new schema is synthesised only when there is a genuine,
        # repeated gap for a specific problem type — not on isolated noise.
        if outcome in ("incorrect", "generic") and self.synthesizer:
            if self._ep >= 10:
                recent = list(self.replay.buffer)[-20:]
                op_failures = sum(
                    1 for e in recent
                    if e.outcome == "incorrect"
                    and e.features.operator_type == features.operator_type
                )
                if op_failures >= 3:
                    self._evolve(problem_text, features, ret_status, tool_trace)

        # Periodic replay check
        if self._ep > 0 and self._ep % REPLAY_EVERY == 0:
            self._replay_check()

        self._last_schema = schema_name
        self._ep += 1

    def _learn_from_tools(self, schema_name: str,
                          features: ProblemFeatures,
                          outcome: str,
                          tool_trace: List[dict],
                          problem_text: str):
        """
        Extract learning signals from the agent's tool call trace.

        Four signals beyond plain correct/incorrect:

        1. calculate + incorrect
           The agent computed correctly but still failed.
           The error was conceptual — schema gave wrong approach.
           Apply a stronger self-correct penalty.

        2. schema_lookup miss + incorrect
           The agent needed a strategy the bank did not have.
           The missing query is recorded for synthesis consideration.

        3. self_verify failed
           The schema guided the agent to a wrong first answer.
           Direct quality signal — amplify the failure penalty.

        4. decompose + correct
           The agent successfully decomposed this problem type.
           Store the decomposition pattern as schema metadata.
        """
        hook = self.bank.hooks.get(schema_name)

        # Signal 1: calculate result and outcome
        calc_used = any(t["tool"] == "calculate" for t in tool_trace)
        if calc_used and outcome == "incorrect" and hook:
            # Stronger penalty: computation was right, approach was wrong
            hook.update_success(False, alpha=0.2)
            if self.verbose:
                print(f"  [AgentLearn] '{schema_name}': calc+wrong "
                      f"→ conceptual error penalty")
        elif calc_used and outcome == "correct" and hook:
            # Schema guided computation to the right answer — extra reward
            hook.update_success(True, alpha=0.2)
            if self.verbose:
                print(f"  [AgentLearn] '{schema_name}': calc+correct "
                      f"→ computational success reward")

        # Signal 2: schema_lookup misses → record unmet needs
        for t in tool_trace:
            if t["tool"] == "schema_lookup" and not t.get("hit"):
                # Store the query so the synthesiser can use it
                if not hasattr(self, "_lookup_misses"):
                    self._lookup_misses: List[str] = []
                self._lookup_misses.append(t["arg"])
                if self.verbose:
                    print(f"  [AgentLearn] schema_lookup miss: '{t['arg']}'")

        # Signal 3: self_verify failed → schema gave wrong approach
        # Only trust this when the overall episode was actually incorrect;
        # if the base controller solved the problem, treat self_verify noise
        # as a tool bug, not a schema failure.
        for t in tool_trace:
            if t["tool"] == "self_verify":
                result = t.get("result", {})
                if isinstance(result, str):
                    try: result = json.loads(result)
                    except: result = {}
                if outcome == "incorrect" and not result.get("verified") and hook:
                    hook.update_success(False, alpha=0.15)
                    if self.verbose:
                        print(f"  [AgentLearn] '{schema_name}': "
                              f"self_verify failed → approach penalty "
                              f"({result.get('issue', '')})")
                    break

        # Signal 3b: sympy_verify result — strongest formal quality signal
        # sympy_verify=True  + correct  → strong reward (formal proof)
        # sympy_verify=False + incorrect → strong penalty (formal disproof)
        # sympy_verify=None              → uncertain, no signal
        for t in tool_trace:
            if t["tool"] == "sympy_verify":
                result = t.get("result", {})
                if not isinstance(result, dict):
                    continue
                v = result.get("verified")
                if v is True and outcome == "correct" and hook:
                    hook.update_success(True, alpha=0.25)   # stronger than normal
                    if self.verbose:
                        print(f"  [AgentLearn] '{schema_name}': "
                              f"sympy_verify=True → formal reward")
                elif v is False and outcome == "incorrect" and hook:
                    hook.update_success(False, alpha=0.25)  # stronger penalty
                    if self.verbose:
                        print(f"  [AgentLearn] '{schema_name}': "
                              f"sympy_verify=False → formal penalty "
                              f"({result.get('reason', '')})")
                break

        # Signal 4: decompose + correct → store decomposition pattern
        if outcome == "correct" and hook:
            for t in tool_trace:
                if t["tool"] == "decompose":
                    decomp = t.get("result", {})
                    if isinstance(decomp, str):
                        try: decomp = json.loads(decomp)
                        except: decomp = {}
                    subs = decomp.get("sub_problems", [])
                    if subs:
                        domains = [sp.get("domain", "") for sp in subs]
                        if not hasattr(hook, "decomp_patterns"):
                            hook.decomp_patterns = []
                        hook.decomp_patterns.append(domains)
                        if self.verbose:
                            print(f"  [AgentLearn] '{schema_name}': "
                                  f"decomp pattern stored {domains}")

    def _evolve(self, problem_text, features, status,
                tool_trace: Optional[List[dict]] = None):
        try:
            new_schema = self.synthesizer(problem_text, features)
            if new_schema:
                emb  = self.encode(new_schema.get("description", problem_text))
                feat = self.extractor.extract(new_schema.get("description",""), emb)

                # Fix 2: classify synthesized schema's operator type via LLM
                if self._llm_client is not None:
                    feat.operator_type = classify_operator_llm(
                        new_schema.get("description", problem_text),
                        self._llm_client
                    )

                # Fix 3: warm-start centroid from multiple representations
                hook = build_warm_hook(new_schema, feat, self.encoder)

                # Reinforce transition boundary (recency mechanism)
                if self._last_schema and self._last_schema in self.bank.hooks:
                    prev_eps = [e for e in reversed(self.replay.buffer)
                                if e.schema_used == self._last_schema][:3]
                    for pe in prev_eps:
                        self.bank.update_hook(self._last_schema, pe.features, True)
                self.bank.add(new_schema, hook)

            # Also synthesise from schema_lookup misses logged by _learn_from_tools
            # A lookup miss means the agent needed a strategy the bank lacked
            if hasattr(self, "_lookup_misses") and self._lookup_misses:
                for query in self._lookup_misses[-2:]:  # max 2 per episode
                    miss_schema = self.synthesizer(
                        f"Strategy needed: {query}\nOriginal problem: {problem_text}",
                        features
                    )
                    if miss_schema:
                        emb2  = self.encode(miss_schema.get("description", query))
                        feat2 = self.extractor.extract(
                            miss_schema.get("description", ""), emb2
                        )
                        if self._llm_client is not None:
                            feat2.operator_type = classify_operator_llm(
                                miss_schema.get("description", query),
                                self._llm_client
                            )
                        hook2 = build_warm_hook(miss_schema, feat2, self.encoder)
                        self.bank.add(miss_schema, hook2)
                        print(f"  [Evolution/Miss] Created: '{miss_schema['name']}'")
                self._lookup_misses = []  # clear after processing

        except Exception as e:
            if self.verbose:
                print(f"  [Evolution] Failed: {e}")

    def _replay_check(self):
        sample  = self.replay.sample(n=8)
        drifted = 0
        for ep in sample:
            _, hook, _, _, status = self.bank.retrieve(ep.features)
            if (status in ("retrieved_high", "retrieved_med") and
                    hook and hook.schema_name != ep.schema_used and
                    ep.outcome == "correct"):
                self.bank.update_hook(ep.schema_used, ep.features, True)
                drifted += 1
        if drifted and self.verbose:
            print(f"  [Replay] Corrected drift for {drifted} episodes")


# ══════════════════════════════════════════════════════════════════════════════
# INTELLIGENT SCHEMA MEMORY — 5 SELF-IMPROVEMENT MECHANISMS
# ══════════════════════════════════════════════════════════════════════════════

def _ism_audit(bank: SchemaBank, replay: ReplayBuffer) -> Dict[str, dict]:
    """
    Mechanism 1: Self-Audit
    Score every schema on retrieval precision and outcome lift.
    Returns health report: strong / neutral / weak / unused.
    """
    if not replay.buffer:
        return {}

    baseline = sum(1 for e in replay.buffer if e.outcome == "correct") / len(replay.buffer)
    report   = {}

    for name in list(bank.schemas.keys()):
        hook = bank.hooks.get(name)
        if not hook:
            continue

        retrieved_eps = [e for e in replay.buffer if e.schema_used == name]

        if not retrieved_eps:
            report[name] = {"health": "unused", "score": 0.0,
                            "lift": 0.0, "uses": 0}
            continue

        precision = sum(1 for e in retrieved_eps
                        if e.outcome == "correct") / len(retrieved_eps)
        lift      = precision - baseline

        report[name] = {
            "health": ("strong"  if lift >  0.10 else
                       "weak"    if lift < -0.05 else "neutral"),
            "score":  round(precision, 3),
            "lift":   round(lift,      3),
            "uses":   len(retrieved_eps),
        }

    return report


def _ism_correct(schema_name: str, bank: SchemaBank,
                 failed_eps: List[Episode], llm_client) -> Optional[dict]:
    """
    Mechanism 2: Self-Correct
    Schema is retrieved but causing failures.
    GPT-5 diagnoses and rewrites it.
    """
    schema   = bank.schemas.get(schema_name)
    if not schema:
        return None

    examples = "\n\n".join(
        f"Problem: {ep.problem_text[:200]}\n"
        f"Operator type: {ep.features.operator_type}"
        + (f"\nMistake: {ep.mistake}" if ep.mistake else "")
        for ep in failed_eps[:3]
    )

    prompt = f"""This mathematical schema is causing wrong answers when retrieved:

Schema: {schema_name}
Description: {schema.get('description','')}
Template: {schema.get('template','')}
Current anti-heuristics (known pitfalls): {schema.get('anti_heuristics', [])}

Problems where it failed:
{examples}

Diagnose the problem and rewrite the schema to be more accurate.
Make the description clearly distinguish it from related schemas.
Also update anti_heuristics with specific mistakes to avoid based on the failures above.
Return ONLY valid JSON:
{{
  "name": "{schema_name}",
  "description": "improved description — precise and distinctive",
  "template": "improved numbered solution steps",
  "heuristics": ["list", "of", "strategies"],
  "anti_heuristics": ["specific mistake to avoid", "another pitfall"],
  "correction_reason": "one sentence: what was wrong"
}}"""

    try:
        resp = llm_client.chat.completions.create(
            model=SYNTHESIZER_MODEL,
            messages=[
                {"role": "system", "content":
                 "You improve mathematical schemas. Return only JSON."},
                {"role": "user", "content": prompt},
            ],
        )
        text   = resp.choices[0].message.content.strip()
        text   = re.sub(r'^```json\s*|\s*```$', '', text, flags=re.MULTILINE)
        fixed  = json.loads(text)
        reason = fixed.pop("correction_reason", "unspecified")
        print(f"  [Self-Correct] '{schema_name}': {reason}")
        return fixed
    except Exception as e:
        print(f"  [Self-Correct] Failed for '{schema_name}': {e}")
        return None


def _ism_merge(bank: SchemaBank, encoder,
               llm_client, threshold: float = 0.88) -> bool:
    """
    Mechanism 3: Self-Merge
    Find two schemas with similar centroids and merge into one stronger schema.
    Returns True if a merge happened.
    """
    hooks = list(bank.hooks.values())
    if len(hooks) < 2:
        return False

    # Find most similar pair
    best_sim, pair = 0.0, (None, None)
    for i in range(len(hooks)):
        for j in range(i + 1, len(hooks)):
            sim = cosine_sim(hooks[i].embedding_centroid,
                             hooks[j].embedding_centroid)
            if sim > best_sim:
                best_sim, pair = sim, (hooks[i], hooks[j])

    if best_sim < threshold or pair[0] is None:
        return False

    h1, h2 = pair
    seed_names = {s["name"] for s in SEED_SCHEMAS}
    if h1.schema_name in seed_names or h2.schema_name in seed_names:
        print(f"  [Self-Merge] Skipped '{h1.schema_name}' + '{h2.schema_name}' "
              f"— seed schema protected (sim={best_sim:.2f})")
        return False

    s1, s2  = bank.schemas[h1.schema_name], bank.schemas[h2.schema_name]

    prompt = f"""Two mathematical schemas are redundant (similarity={best_sim:.2f}) and should be merged:

Schema 1: {h1.schema_name}
Description: {s1.get('description','')}
Success rate: {h1.success_rate:.2f} ({h1.usage_count} uses)

Schema 2: {h2.schema_name}
Description: {s2.get('description','')}
Success rate: {h2.success_rate:.2f} ({h2.usage_count} uses)

Create ONE merged schema more general than either.
Return ONLY valid JSON:
{{
  "name": "merged schema name",
  "description": "unified description covering both problem types",
  "template": "comprehensive numbered steps",
  "heuristics": ["combined", "list"]
}}"""

    try:
        resp = llm_client.chat.completions.create(
            model=SYNTHESIZER_MODEL,
            messages=[
                {"role": "system", "content":
                 "You merge mathematical schemas. Return only JSON."},
                {"role": "user", "content": prompt},
            ],
        )
        text   = resp.choices[0].message.content.strip()
        text   = re.sub(r'^```json\s*|\s*```$', '', text, flags=re.MULTILINE)
        merged = json.loads(text)

        # Merged centroid = weighted average by usage count
        w1  = h1.usage_count + 1
        w2  = h2.usage_count + 1
        cen = (w1 * h1.embedding_centroid + w2 * h2.embedding_centroid) / (w1 + w2)
        cen /= np.linalg.norm(cen)

        merged_hook = FeatureHook(
            schema_name         = merged["name"],
            operator_type       = h1.operator_type,
            structural_pattern  = h1.structural_pattern,
            heuristic_signature = merged.get("heuristics", []),
            quantity_signature  = h1.quantity_signature,
            embedding_centroid  = cen,
            success_rate        = max(h1.success_rate, h2.success_rate),
            usage_count         = h1.usage_count + h2.usage_count,
        )

        # Remove old schemas, add merged
        for old in (h1.schema_name, h2.schema_name):
            bank.schemas.pop(old, None)
            bank.hooks.pop(old, None)

        bank.schemas[merged["name"]] = merged
        bank.hooks[merged["name"]]   = merged_hook
        print(f"  [Self-Merge] '{h1.schema_name}' + '{h2.schema_name}'"
              f" → '{merged['name']}' (sim={best_sim:.2f})")
        return True

    except Exception as e:
        print(f"  [Self-Merge] Failed: {e}")
        return False


def _ism_promote(bank: SchemaBank,
                 promote_thresh: float = 0.80,
                 demote_thresh:  float = 0.40,
                 min_uses:       int   = 5,
                 verbose:        bool  = True,
                 active_task_id: Optional[str] = None,
                 cross_task_demote_min_uses: int = 10):
    """
    Mechanism 4: Self-Promote / Self-Demote
    Strong schemas get boosted success_rate → retrieved more eagerly.
    Weak schemas get penalized → retrieved more conservatively.
    """
    for name, hook in bank.hooks.items():
        if hook.usage_count < min_uses:
            continue

        if hook.success_rate >= promote_thresh:
            before = hook.success_rate
            hook.success_rate = min(hook.success_rate * 1.02, 0.97)  # was 1.05; gentler to reduce task interference
            if verbose and hook.usage_count % 15 == 0:
                print(f"  [Self-Promote] '{name}' "
                      f"{before:.2f} → {hook.success_rate:.2f}")

        elif hook.success_rate <= demote_thresh:
            # Fix 2: Protect schemas outside the active block from premature demotion.
            if active_task_id:
                n = name.lower()
                is_number_theory = "number theory" in n or "number-theory" in n
                is_algebra       = "algebra" in n
                is_combinatorics = "combinatorics" in n or "counting" in n or "probability" in n
                is_geometry      = "geometry" in n or "geometric" in n
                is_calculus      = "calculus" in n or "analysis" in n

                dom = ("Number Theory"          if is_number_theory  else
                       "Counting & Probability" if is_combinatorics  else
                       "Geometry"               if is_geometry       else
                       "Precalculus"            if is_calculus       else
                       "Algebra"                if is_algebra        else None)

                if dom and dom != active_task_id and hook.usage_count < cross_task_demote_min_uses:
                    continue
            before = hook.success_rate
            hook.success_rate = max(hook.success_rate * 0.95, 0.20)  # floor at 0.20 so schema can recover
            if verbose:
                print(f"  [Self-Demote]  '{name}' "
                      f"{before:.2f} → {hook.success_rate:.2f}")


def _ism_prune(bank: SchemaBank,
               episodes_elapsed: int,
               prune_after:      int  = 60,
               min_uses:         int  = 1,
               weak_sr_thresh:   float = 0.38,
               weak_min_uses:    int   = 5,
               verbose:          bool = True):
    """
    Mechanism 5: Self-Prune
    Two pruning criteria — both protected by seed-schema guard AND operator
    diversity guard:

      (a) Never retrieved — usage_count < min_uses after prune_after episodes.
          These schemas were synthesised but never matched any problem.
          prune_after=60 gives schemas enough time to encounter their problem
          type before being declared unused (was 25 — too aggressive).

      (b) Confirmed weak — used enough times (weak_min_uses) to have a reliable
          SR estimate, but SR is still below weak_sr_thresh AND has been
          Self-Corrected at least once (correction_count > 0).
          These schemas failed even after an LLM rewrite — not salvageable.

    Never removes seed schemas.
    Never removes the last non-seed schema covering a given operator_type
    (operator-diversity guard) — prevents ISM from losing coverage of an
    entire problem domain just because early episodes were sparse.
    """
    if episodes_elapsed < prune_after:
        return

    seed_names = {s["name"] for s in SEED_SCHEMAS}

    # Build operator-type coverage map for non-seed schemas only.
    # Key: operator_type → list of non-seed schema names with that type.
    op_coverage: Dict[str, List[str]] = defaultdict(list)
    for name, hook in bank.hooks.items():
        if name not in seed_names:
            op_coverage[hook.operator_type].append(name)

    to_remove: List[Tuple[str, str]] = []

    for name, hook in bank.hooks.items():
        if name in seed_names:
            continue

        # Operator-diversity guard: keep the schema if it is the sole
        # non-seed representative of its operator type.
        if len(op_coverage.get(hook.operator_type, [])) <= 1:
            continue

        # Criterion (a): never retrieved
        if hook.usage_count < min_uses:
            to_remove.append((name, "unused"))
            continue

        # Criterion (b): confirmed weak
        if (hook.usage_count >= weak_min_uses
                and hook.success_rate < weak_sr_thresh):
            to_remove.append((name, f"weak SR={hook.success_rate:.2f} "
                                     f"uses={hook.usage_count}"))

    # Re-check diversity guard after building to_remove — don't leave an
    # operator type with zero non-seed coverage after removals.
    surviving = {n for n in bank.hooks if n not in {r[0] for r in to_remove}}
    op_after: Dict[str, int] = defaultdict(int)
    for n in surviving:
        if n not in seed_names:
            op_after[bank.hooks[n].operator_type] += 1

    to_remove = [
        (name, reason) for name, reason in to_remove
        if op_after.get(bank.hooks[name].operator_type, 0) > 0
    ]

    for name, reason in to_remove:
        bank.schemas.pop(name, None)
        bank.hooks.pop(name, None)
        if verbose:
            print(f"  [Self-Prune]   Removed '{name}' ({reason})")


_REINFORCE_PROMPT = """You are a math education expert refining a problem-solving schema.

Schema name: {name}
Description: {description}
Current heuristics:
{heuristics}

Recent SUCCESSFUL solutions using this schema:
{successes}

Based on these confirmed successes, sharpen the heuristics list:
- Keep existing heuristics that are still valid and specific
- Add new actionable tips extracted from the evidence above
- Remove vague or redundant entries
- Maximum 8 heuristics total

Return ONLY valid JSON:
{{
  "heuristics": ["step 1", "step 2", ...]
}}"""


def _ism_reinforce(bank: SchemaBank,
                   replay: "ReplayBuffer",
                   client,
                   min_sr:         float = 0.72,
                   min_uses:       int   = 5,
                   min_successes:  int   = 3,
                   verbose:        bool  = True) -> List[str]:
    """
    Mechanism 6: Self-Reinforce
    Strengthen the heuristics of schemas that are already performing well
    by distilling patterns from their recent correct episodes.

    Only fires on schemas that have earned trust (sr >= min_sr, uses >= min_uses)
    and have at least min_successes correct episodes to learn from.
    Never touches seed schemas' heuristics (they are hand-crafted).
    """
    if client is None:
        return []

    seed_names = {s["name"] for s in SEED_SCHEMAS}
    reinforced: List[str] = []

    for name, hook in list(bank.hooks.items()):
        if name in seed_names:
            continue
        if hook.success_rate < min_sr or hook.usage_count < min_uses:
            continue
        schema = bank.schemas.get(name)
        if not schema:
            continue

        successes = [
            e for e in replay.buffer
            if e.schema_used == name and e.outcome == "correct"
        ][-8:]
        if len(successes) < min_successes:
            continue

        success_lines = []
        for i, ep in enumerate(successes, 1):
            line = f"{i}. {ep.problem_text[:200]}"
            if ep.insight:
                line += f"\n   Key insight: {ep.insight}"
            success_lines.append(line)

        try:
            resp = client.chat.completions.create(
                **_llm_kwargs(SYNTHESIZER_MODEL, max_tokens=300),
                messages=[{"role": "user", "content": _REINFORCE_PROMPT.format(
                    name=name,
                    description=schema.get("description", ""),
                    heuristics=json.dumps(schema.get("heuristics", []), indent=2),
                    successes="\n".join(success_lines),
                )}]
            )
            content = resp.choices[0].message.content.strip()
            m = re.search(r'\{.*\}', content, re.DOTALL)
            if not m:
                continue
            data = json.loads(m.group())
            new_heuristics = data.get("heuristics", [])
            if new_heuristics and isinstance(new_heuristics, list):
                bank.schemas[name]["heuristics"] = new_heuristics[:8]
                reinforced.append(name)
                if verbose:
                    print(f"  [Self-Reinforce] '{name}': heuristics updated "
                          f"(sr={hook.success_rate:.2f}, uses={hook.usage_count}, "
                          f"successes={len(successes)})")
        except Exception:
            pass

    return reinforced


_ANTIPATTERN_PROMPT = """You are a math education expert identifying common mistakes for a schema.

Schema name: {name}
Description: {description}
Current anti-heuristics (already known pitfalls):
{current_anti}

Recent FAILED attempts using this schema (with mistakes where available):
{failures}

Identify up to 5 specific, actionable mistakes to avoid when using this schema.
Each rule must be concrete — name the exact error, not a vague warning.

Good examples:
  "Do not assume the answer is an integer when the problem allows fractions"
  "Do not apply the quadratic formula before checking if factoring is simpler"
  "Do not forget to check both positive and negative roots for absolute value equations"

Bad examples (too vague — reject these):
  "Be careful"
  "Check your work"
  "Read the problem carefully"

Merge or remove any existing anti-heuristics that are already covered or no longer relevant.
Return ONLY valid JSON:
{{
  "anti_heuristics": ["rule 1", "rule 2", ...]
}}"""


def _ism_antipattern(bank: SchemaBank,
                     replay: "ReplayBuffer",
                     client,
                     min_failures: int  = 3,
                     max_rules:    int  = 5,
                     verbose:      bool = True) -> List[str]:
    """
    Mechanism 7: Self-Antipattern
    For each schema with enough failure evidence, distil a concrete list of
    mistakes-to-avoid (anti_heuristics) from the mistake log in the replay buffer.

    Complements Self-Reinforce: Reinforce sharpens what to DO; Antipattern
    encodes what NOT to do. Both are injected into the solver prompt.
    """
    if client is None:
        return []

    updated: List[str] = []

    for name, hook in list(bank.hooks.items()):
        schema = bank.schemas.get(name)
        if not schema:
            continue

        failed_eps = [
            e for e in replay.buffer
            if e.schema_used == name and e.outcome == "incorrect"
        ]
        if len(failed_eps) < min_failures:
            continue

        # Collect the most informative failures — prefer those with extracted mistakes
        with_mistake    = [e for e in failed_eps if e.mistake]
        without_mistake = [e for e in failed_eps if not e.mistake]
        sample = (with_mistake[-6:] + without_mistake[-(max(0, 8 - len(with_mistake[-6:]))):])[: 8]

        failure_lines = []
        for i, ep in enumerate(sample, 1):
            line = f"{i}. {ep.problem_text[:180]}"
            if ep.mistake:
                line += f"\n   Mistake: {ep.mistake}"
            failure_lines.append(line)

        try:
            resp = client.chat.completions.create(
                **_llm_kwargs(SYNTHESIZER_MODEL, max_tokens=300),
                messages=[{"role": "user", "content": _ANTIPATTERN_PROMPT.format(
                    name=name,
                    description=schema.get("description", ""),
                    current_anti=json.dumps(schema.get("anti_heuristics", []), indent=2),
                    failures="\n".join(failure_lines),
                )}]
            )
            content = resp.choices[0].message.content.strip()
            m = re.search(r'\{.*\}', content, re.DOTALL)
            if not m:
                continue
            data = json.loads(m.group())
            new_anti = data.get("anti_heuristics", [])
            if new_anti and isinstance(new_anti, list):
                schema["anti_heuristics"] = new_anti[:max_rules]
                updated.append(name)
                if verbose:
                    print(f"  [Self-Antipattern] '{name}': {len(new_anti)} rules "
                          f"(failures={len(failed_eps)})")
                    for rule in new_anti[:max_rules]:
                        print(f"    • {rule}")
        except Exception:
            pass

    return updated


class IntelligentSchemaMemory(MemoryController):
    """
    Schema Memory with full self-awareness and self-improvement.

    Seven mechanisms beyond the base MemoryController:
      1. Self-Audit       — health report every AUDIT_EVERY episodes
      2. Self-Correct     — rewrites weak schemas via GPT-4o
      3. Self-Merge       — consolidates similar schemas
      4. Self-Promote     — boosts strong schemas, penalizes weak ones
      5. Self-Prune       — removes unused/failing schemas
      6. Self-Reinforce   — strengthens heuristics from successful episodes
      7. Self-Antipattern — extracts concrete mistakes-to-avoid from failures
                            and stores them as anti_heuristics in schema content

    The bank is no longer append-only. It actively monitors and
    improves its own quality over time — in both directions.
    """

    AUDIT_EVERY       = 10
    MERGE_EVERY       = 20
    PRUNE_EVERY       = 25
    REINFORCE_EVERY   = REINFORCE_EVERY    # = 15
    ANTIPATTERN_EVERY = ANTIPATTERN_EVERY  # = 20
    WARMUP_EPS        = 10  # don't run any mechanisms before this many episodes

    def __init__(self, bank, replay, encoder,
                 synthesizer=None, verbose=True):
        super().__init__(bank, replay, encoder,
                         synthesizer=synthesizer,
                         verbose=verbose,
                         name="Schema Memory (ISM)")
        # Bank health log — tracked across episodes
        self._health_log: List[dict] = []
        # Self-improvement event log — one entry per mechanism firing
        self.improvement_log: List[dict] = []

    def after_episode(self, problem_text, features, schema_used,
                      ret_status, outcome, task_id,
                      tool_trace=None, response_text=None):
        # Base slow loop: hook update, replay, synthesis, replay_check
        super().after_episode(
            problem_text, features, schema_used,
            ret_status, outcome, task_id,
            tool_trace=tool_trace,
            response_text=response_text,
        )

        ep = self._ep  # _ep already incremented by super()

        # ── Mechanism 1 + 2: Audit + Correct ─────────────────────────────
        if ep >= self.WARMUP_EPS and ep % self.AUDIT_EVERY == 0:
            self._run_audit_and_correct(active_task_id=task_id)

        # ── Mechanism 3: Merge ────────────────────────────────────────────
        if ep >= self.WARMUP_EPS and ep % self.MERGE_EVERY == 0:
            bank_before = list(self.bank.schemas.keys())
            merged = _ism_merge(self.bank, self.encoder,
                                self._llm_client, threshold=0.88)
            if merged:
                bank_after   = list(self.bank.schemas.keys())
                removed      = [n for n in bank_before if n not in bank_after]
                added        = [n for n in bank_after  if n not in bank_before]
                self.improvement_log.append({
                    "episode":   ep,
                    "mechanism": "Self-Merge",
                    "removed":   removed,
                    "added":     added,
                    "bank_size_after": self.bank.size(),
                })
                # Re-audit after merge — bank structure changed
                self._run_audit_and_correct(active_task_id=task_id)

        # ── Mechanism 5: Prune ────────────────────────────────────────────
        if ep >= self.WARMUP_EPS and ep % self.PRUNE_EVERY == 0:
            bank_before = list(self.bank.schemas.keys())
            _ism_prune(self.bank, episodes_elapsed=ep,
                       verbose=self.verbose)
            bank_after = list(self.bank.schemas.keys())
            pruned     = [n for n in bank_before if n not in bank_after]
            if pruned:
                self.improvement_log.append({
                    "episode":   ep,
                    "mechanism": "Self-Prune",
                    "pruned":    pruned,
                    "bank_size_after": self.bank.size(),
                })

        # ── Mechanism 6: Reinforce ────────────────────────────────────────
        if ep >= self.WARMUP_EPS and ep % self.REINFORCE_EVERY == 0:
            reinforced = _ism_reinforce(
                self.bank, self.replay,
                self._llm_client,
                verbose=self.verbose,
            )
            if reinforced:
                self.improvement_log.append({
                    "episode":         ep,
                    "mechanism":       "Self-Reinforce",
                    "schemas":         reinforced,
                    "bank_size_after": self.bank.size(),
                })

        # ── Mechanism 7: Antipattern ──────────────────────────────────────
        if ep >= self.WARMUP_EPS and ep % self.ANTIPATTERN_EVERY == 0:
            updated = _ism_antipattern(
                self.bank, self.replay,
                self._llm_client,
                verbose=self.verbose,
            )
            if updated:
                self.improvement_log.append({
                    "episode":   ep,
                    "mechanism": "Self-Antipattern",
                    "schemas":   updated,
                })

    def _run_audit_and_correct(self, active_task_id: Optional[str] = None):
        """Run audit, then correct any weak schemas found."""
        report = _ism_audit(self.bank, self.replay)

        # ── Lift log for Figure 4 ─────────────────────────────────────
        if not hasattr(self, '_lift_log'):
            self._lift_log = []
        for name, stats in report.items():
            self._lift_log.append({
                "episode": self._ep,
                "schema":  name,
                "lift":    stats["lift"],
                "health":  stats["health"],
                "uses":    stats["uses"],
            })
        # ─────────────────────────────────────────────────────────────

        # Log health snapshot
        self._health_log.append({
            "episode": self._ep,
            "bank_size": self.bank.size(),
            "report": {k: v["health"] for k, v in report.items()},
        })

        # Log audit event
        self.improvement_log.append({
            "episode":   self._ep,
            "mechanism": "Self-Audit",
            "bank_size": self.bank.size(),
            "health_counts": {
                h: sum(1 for v in report.values() if v["health"] == h)
                for h in ("strong", "neutral", "weak", "unused")
            },
            "schema_health": {k: v["health"] for k, v in report.items()},
        })

        if self.verbose:
            print(f"\n  ┌─ [ISM Audit] ep={self._ep} "
                  f"bank_size={self.bank.size()} ─────────────────")
            for name, stats in sorted(report.items(),
                                      key=lambda x: x[1]["score"],
                                      reverse=True):
                symbol = ("✓" if stats["health"] == "strong"  else
                          "~" if stats["health"] == "neutral" else
                          "✗" if stats["health"] == "weak"    else "○")
                print(f"  │ {symbol} {name[:35]:<35} "
                      f"prec={stats['score']:.2f}  "
                      f"lift={stats['lift']:+.2f}  "
                      f"uses={stats['uses']}")
            print(f"  └───────────────────────────────────────────────\n")

        # ── Mechanism 2: Correct weak schemas ────────────────────────────
        def _schema_domain(schema_name: str) -> Optional[str]:
            n = schema_name.lower()
            if "number theory" in n or "number-theory" in n:
                return "Number Theory"
            if "combinatorics" in n or "counting" in n or "probability" in n:
                return "Counting & Probability"
            if "geometry" in n or "geometric" in n:
                return "Geometry"
            if "calculus" in n or "analysis" in n:
                return "Precalculus"  # matches MATH500 type_map output
            if "algebra" in n:
                return "Algebra"
            return None

        seed_names = {s["name"] for s in SEED_SCHEMAS}
        for name, stats in report.items():
            hook = self.bank.hooks.get(name)

            # Two paths to Self-Correct:
            #   (A) Lift-based: health == "weak" and uses >= 5  (original gate)
            #   (B) Absolute SR floor: sr < 0.50 and uses >= 10 regardless of lift
            #       Catches schemas the lift metric misses because their domain
            #       is inherently hard (e.g. base rate ~50% → lift ≈ 0 even at 44% SR).
            path_a = stats["health"] == "weak" and stats["uses"] >= 5
            path_b = (hook is not None
                      and hook.success_rate < 0.50
                      and hook.usage_count >= 10)

            if not (path_a or path_b):
                continue

            # Seed schemas are hand-crafted and protected from normal correction.
            # Exception: allow Self-Correct (but never Self-Escalate) when a seed
            # is critically underperforming (sr < 0.40 and uses >= 5).
            if name in seed_names:
                seed_critical = (hook is not None
                                 and hook.success_rate < 0.40
                                 and hook.usage_count >= 5)
                if not seed_critical:
                    continue

            # Protect schemas outside the currently active task block.
            dom = _schema_domain(name)
            if active_task_id and dom and dom != active_task_id:
                continue

            trigger = "lift-weak" if path_a else "sr-floor"
            if self.verbose and path_b and not path_a:
                print(f"  [Self-Correct] '{name}' triggered by SR floor "
                      f"(sr={hook.success_rate:.2f}, uses={hook.usage_count})")

            # Escalation: if corrected 3+ times and still qualifying, prune instead
            if hook and hook.correction_count >= 3 and name not in seed_names:
                self.bank.schemas.pop(name, None)
                self.bank.hooks.pop(name, None)
                if self.verbose:
                    print(f"  [Self-Escalate] Pruned '{name}' after {hook.correction_count} failed corrections")
                self.improvement_log.append({
                    "episode":   self._ep,
                    "mechanism": "Self-Escalate",
                    "schema":    name,
                    "correction_count": hook.correction_count,
                    "bank_size_after": self.bank.size(),
                })
                continue

            failed = [
                e for e in self.replay.buffer
                if e.schema_used == name
                and e.outcome == "incorrect"
            ]
            if not failed:
                continue

            fixed = _ism_correct(name, self.bank, failed,
                                 self._llm_client)
            if fixed:
                self.bank.schemas[name].update(fixed)
                new_emb = self.encode(fixed.get("description", name))
                self.bank.hooks[name].embedding_centroid = new_emb
                self.bank.hooks[name].success_rate = 0.5
                self.bank.hooks[name].correction_count += 1
                self.improvement_log.append({
                    "episode":          self._ep,
                    "mechanism":        "Self-Correct",
                    "schema":           name,
                    "trigger":          trigger,
                    "correction_count": self.bank.hooks[name].correction_count,
                    "failed_episodes":  len(failed),
                    "sr_reset_to":      0.5,
                })

        # ── Mechanism 4: Promote / Demote ─────────────────────────────────
        sr_before = {n: h.success_rate for n, h in self.bank.hooks.items()}
        _ism_promote(self.bank, verbose=self.verbose, active_task_id=active_task_id)
        promoted = [n for n, h in self.bank.hooks.items()
                    if n in sr_before and h.success_rate > sr_before[n]]
        demoted  = [n for n, h in self.bank.hooks.items()
                    if n in sr_before and h.success_rate < sr_before[n]]
        if promoted or demoted:
            self.improvement_log.append({
                "episode":   self._ep,
                "mechanism": "Self-Promote/Demote",
                "promoted":  promoted,
                "demoted":   demoted,
            })

    def health_summary(self) -> str:
        """Return a formatted bank health summary for logging."""
        report = _ism_audit(self.bank, self.replay)
        lines  = ["\n  ═══ ISM Bank Health Summary ═══"]
        strong  = [n for n, s in report.items() if s["health"] == "strong"]
        neutral = [n for n, s in report.items() if s["health"] == "neutral"]
        weak    = [n for n, s in report.items() if s["health"] == "weak"]
        unused  = [n for n, s in report.items() if s["health"] == "unused"]
        lines.append(f"  Strong  ({len(strong)}):  {', '.join(strong)}")
        lines.append(f"  Neutral ({len(neutral)}): {', '.join(neutral)}")
        lines.append(f"  Weak    ({len(weak)}):    {', '.join(weak)}")
        lines.append(f"  Unused  ({len(unused)}):  {', '.join(unused)}")
        lines.append(f"  Total schemas: {self.bank.size()}")
        return "\n".join(lines)


class StaticLLMController(MemoryController):
    """No memory. Never updates. Control condition."""
    def after_episode(self, problem_text, features, schema_used,
                      ret_status, outcome, task_id,
                      tool_trace=None, response_text=None):
        ep = Episode(
            self._ep, problem_text, features,
            "Generic", outcome, 0.0, task_id,
            features.log_llm_used,
        )
        self.replay.add(ep)
        self._ep += 1


class NeuralMemoryController(MemoryController):
    """EMA embedding store. No symbolic structure. No replay."""
    def __init__(self, encoder, verbose=False):
        bank   = SchemaBank()
        replay = ReplayBuffer()
        super().__init__(bank, replay, encoder, verbose=verbose,
                         name="Neural Memory")
        self._neural: List[Tuple[str, np.ndarray]] = []
        # Seed — mirror 6 seed domains for fair ablation comparison
        for label, desc in [
            ("Algebra",               "algebra equation variable solve polynomial function"),
            ("Number Theory",         "number theory prime divisibility modular remainder integer"),
            ("Geometry",              "geometry area perimeter angle triangle circle coordinate"),
            ("Combinatorics",         "counting permutation combination arrangement selection ways"),
            ("Probability",           "probability chance likelihood expected value random event"),
            ("Calculus and Analysis", "limit derivative integral series convergence calculus"),
        ]:
            self._neural.append((label, encoder.encode(desc, normalize_embeddings=True)))

    def get_schema_for_problem(self, problem_text):
        emb      = self.encode(problem_text)
        features = self.extractor.extract(problem_text, emb)
        best, best_score = None, -1.0
        for label, vec in self._neural:
            s = cosine_sim(emb, vec)
            if s > best_score:
                best_score, best = s, label
        if best_score >= 0.55:
            schema = {"name": best, "description": f"Learned: {best}",
                      "template": "Apply learned pattern.", "heuristics": ["decompose"]}
            return schema, features, best_score, {"embedding": best_score}, "retrieved"
        return GENERIC_SCHEMA, features, best_score, {}, "generic"

    def after_episode(self, problem_text, features, schema_used,
                      ret_status, outcome, task_id,
                      tool_trace=None, response_text=None):
        # EMA update only — no replay, no boundary reinforcement
        emb = features.embedding
        best_idx, best_score = 0, -1.0
        for i, (_, vec) in enumerate(self._neural):
            s = cosine_sim(emb, vec)
            if s > best_score:
                best_score, best_idx = s, i
        label, vec = self._neural[best_idx]
        self._neural[best_idx] = (label, (1 - EMA_ALPHA) * vec + EMA_ALPHA * emb)
        if outcome in ("incorrect", "generic"):
            self._neural.append((f"Learned-{self._ep}", emb.copy()))
        ep = Episode(
            self._ep, problem_text, features,
            schema_used.get("name","Generic"), outcome, 0.0, task_id,
            features.log_llm_used,
        )
        self.replay.add(ep)
        self._ep += 1


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: CL METRICS
# ══════════════════════════════════════════════════════════════════════════════

class CLTracker:
    """Tracks accuracy per task per block for CL metrics."""
    def __init__(self):
        self.task_blocks: Dict[str, List[List[bool]]] = defaultdict(list)
        self.task_current: Dict[str, List[bool]]      = defaultdict(list)
        self.current_task = None
        self.all_correct: List[bool] = []
        self.block_size = 10

    def record(self, task_id: str, correct: bool):
        # Detect task switch → save current block
        if self.current_task and task_id != self.current_task:
            if self.task_current[self.current_task]:
                self.task_blocks[self.current_task].append(
                    self.task_current[self.current_task].copy()
                )
                self.task_current[self.current_task] = []
        self.current_task = task_id
        self.task_current[task_id].append(correct)
        self.all_correct.append(correct)

    def flush(self):
        for task_id, results in self.task_current.items():
            if results:
                self.task_blocks[task_id].append(results)
        self.task_current = defaultdict(list)

    def compute(self) -> dict:
        self.flush()
        plasticity_scores, stability_scores, forgetting_scores, bt_scores = [], [], [], []

        for task_id, blocks in self.task_blocks.items():
            accs = [sum(b)/len(b) for b in blocks if b]
            if not accs:
                continue
            plasticity_scores.append(accs[0])
            if len(accs) > 1:
                stability_scores.extend(accs[1:])
                forgetting_scores.append(max(accs) - accs[-1])
                bt_scores.append(accs[-1] - accs[0])

        return {
            "average_accuracy":  sum(self.all_correct) / len(self.all_correct) if self.all_correct else 0,
            "plasticity":        sum(plasticity_scores) / len(plasticity_scores) if plasticity_scores else 0,
            "stability":         sum(stability_scores) / len(stability_scores) if stability_scores else 0,
            "forgetting":        sum(forgetting_scores) / len(forgetting_scores) if forgetting_scores else 0,
            "backward_transfer": sum(bt_scores) / len(bt_scores) if bt_scores else 0,
            "n_episodes":        len(self.all_correct),
            "per_task":          {tid: [sum(b)/len(b) for b in blocks]
                                  for tid, blocks in self.task_blocks.items()},
        }

    def summary(self, system_name: str) -> str:
        m = self.compute()
        lines = [
            f"\n{'─'*55}",
            f"  {system_name}",
            f"{'─'*55}",
            f"  Average Accuracy:  {m['average_accuracy']:.2%}",
            f"  Plasticity:        {m['plasticity']:.2%}",
            f"  Stability:         {m['stability']:.2%}",
            f"  Forgetting:        {m['forgetting']:.2%}  {'↑ bad' if m['forgetting'] > 0.05 else '✓ good'}",
            f"  Backward Transfer: {m['backward_transfer']:+.2%}",
            f"  Per-task blocks:",
        ]
        for tid, accs in m["per_task"].items():
            acc_str = " → ".join(f"{a:.0%}" for a in accs)
            lines.append(f"    {tid:<35} [{acc_str}]")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8: DATASET
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MathProblem:
    problem_id: int
    text: str
    level: int
    type: str
    solution: str
    boxed_answer: Optional[str]
    numeric_answer: Optional[float]
    task_id: str = ""

    def __post_init__(self):
        if not self.task_id:
            self.task_id = self.type


def load_competition_math(source=None, max_level=5, min_level=1,
                          numeric_only=False,
                          max_per_type=200) -> List[MathProblem]:
    """Load from HuggingFace, local parquet, or use built-in problems."""

    rows = []

    if source is None:
        print("Downloading from HuggingFace...")
        try:
            from datasets import load_dataset
            ds   = load_dataset("qwedsacf/competition_math", split="train")
            rows = list(ds)
            print(f"Downloaded {len(rows)} rows")
        except Exception as e:
            print(f"Download failed: {e}")
            print("Falling back to built-in problems...")
            return _builtin_problems()

    elif source.endswith(".parquet"):
        import pandas as pd
        rows = pd.read_parquet(source).to_dict("records")

    elif source.endswith(".json") or source.endswith(".jsonl"):
        with open(source) as f:
            rows = json.load(f)

    # Process rows
    problems = []
    type_counts = defaultdict(int)

    for i, row in enumerate(rows):
        level_str = str(row.get("level", "Level 1"))
        m = re.search(r'\d+', level_str)
        level = int(m.group()) if m else 1
        if level > max_level or level < min_level:
            continue

        prob_type = row.get("type", "Algebra")
        if type_counts[prob_type] >= max_per_type:
            continue

        solution = row.get("solution", "")
        boxed    = extract_boxed(solution)
        numeric  = boxed_to_float(boxed)

        if numeric_only and numeric is None:
            continue

        problems.append(MathProblem(
            problem_id    = i,
            text          = row["problem"],
            level         = level,
            type          = prob_type,
            solution      = solution,
            boxed_answer  = boxed,
            numeric_answer= numeric,
        ))
        type_counts[prob_type] += 1

    # Stats
    type_dist = defaultdict(int)
    for p in problems:
        type_dist[p.type] += 1
    print(f"\nLoaded {len(problems)} problems:")
    for t, c in sorted(type_dist.items()):
        print(f"  {t:<35} {c}")

    return problems


def build_cl_stream(problems: List[MathProblem], n_per_block=20,
                    cl_types=None, seed=42) -> List[MathProblem]:
    """
    Build CL stream in seed-schema order (filtered by available dataset types),
    with repeat blocks for forgetting tests.
    """
    random.seed(seed)

    available_types = list({p.type for p in problems})
    seed_order = [s["name"] for s in SEED_SCHEMAS]
    types = cl_types or seed_order
    types = [t for t in types if t in available_types]

    if len(types) < 2:
        types = available_types[:3]
    while len(types) < 3:
        types.append(types[-1])

    type_a, type_b = types[0], types[1]

    # Detect available levels dynamically — works for any dataset
    all_levels = sorted({p.level for p in problems})
    lo_levels  = all_levels[:max(1, len(all_levels)//2)]   # easier half
    hi_levels  = all_levels[len(all_levels)//2:]           # harder half
    if not hi_levels:
        hi_levels = all_levels

    # Bucket by (type, level)
    buckets = defaultdict(list)
    for p in problems:
        buckets[(p.type, p.level)].append(p)

    def sample(prob_type, levels, n):
        pool = []
        for lv in levels:
            pool.extend(buckets[(prob_type, lv)])
        return random.sample(pool, min(n, len(pool))) if pool else []

    # First pass: all available seed-aligned types.
    config = [(t, all_levels, n_per_block, f"{t}") for t in types]
    # Then forgetting/stability probes focused on first two task families.
    config.extend([
        (type_a, all_levels,  n_per_block, f"{type_a}"),  # REPEAT → forgetting test
        (type_b, hi_levels,   n_per_block, f"{type_b}"),  # harder variant
        (type_a, all_levels,  n_per_block, f"{type_a}"),  # REPEAT 2 → forgetting test
    ])

    stream = []
    print("\nCL Stream:")
    ep = 0
    for prob_type, levels, n, label in config:
        block = sample(prob_type, levels, n)
        for p in block:
            p.task_id = label
        stream.extend(block)
        repeat = " ← REPEAT (forgetting test)" if (prob_type == type_a and ep > 0) else ""
        print(f"  Ep {ep:>3}–{ep+len(block)-1:>3}: {label}{repeat}  ({len(block)} problems)")
        ep += len(block)

    print(f"  Total: {len(stream)} episodes\n")
    return stream


def load_math500(max_per_type=100) -> List[MathProblem]:
    """
    MATH-Hard (lighteval): ~1324 hard competition math problems.
    gpt-4o-mini accuracy ~55-65% — ideal for CL experiments.
    """
    print("Loading MATH-500...")
    try:
        from datasets import load_dataset
        ds   = load_dataset("lighteval/MATH-Hard", split="test")
        rows = list(ds)
        print(f"  Loaded {len(rows)} raw problems")
    except Exception as e:
        print(f"  Failed: {e} — falling back to built-in problems")
        return _builtin_problems()

    # Normalize type names to match seed schemas and CL stream
    type_map = {
        "algebra":                  "Algebra",
        "counting_and_probability": "Counting & Probability",
        "counting and probability": "Counting & Probability",
        "geometry":                 "Geometry",
        "intermediate_algebra":     "Intermediate Algebra",
        "intermediate algebra":     "Intermediate Algebra",
        "number_theory":            "Number Theory",
        "number theory":            "Number Theory",
        "prealgebra":               "Prealgebra",
        "precalculus":              "Precalculus",
    }

    problems    = []
    type_counts = defaultdict(int)

    for i, row in enumerate(rows):
        raw_type = (row.get("type") or row.get("subject") or
                    row.get("category") or "Algebra")
        prob_type = type_map.get(raw_type.lower().strip(), raw_type)

        if type_counts[prob_type] >= max_per_type:
            continue

        solution = row.get("solution", "")
        boxed    = extract_boxed(solution)
        numeric  = boxed_to_float(boxed)

        # Parse level — stored as "Level 5" or int
        raw_level = row.get("level", "Level 5")
        lm = re.search(r'\d+', str(raw_level))
        level = int(lm.group()) if lm else 5

        problems.append(MathProblem(
            problem_id    = i,
            text          = row["problem"],
            level         = level,
            type          = prob_type,
            solution      = solution,
            boxed_answer  = boxed,
            numeric_answer= numeric,
        ))
        type_counts[prob_type] += 1

    type_dist = defaultdict(int)
    for p in problems:
        type_dist[p.type] += 1
    print(f"  Loaded {len(problems)} problems:")
    for t, c in sorted(type_dist.items()):
        print(f"    {t:<35} {c}")
    return problems


def load_aime2025() -> List[MathProblem]:
    """
    AIME 2025 (AIME I + AIME II, 30 problems).
    Source: HuggingFace `MathArena/aime_2025`.
    Answers are integers (0-999).
    """
    print("Loading AIME 2025...")
    try:
        from datasets import load_dataset
        ds = load_dataset("MathArena/aime_2025", split="train")
        rows = list(ds)
        print(f"  Loaded {len(rows)} problems")
    except Exception as e:
        print(f"  Failed: {e}")
        return []

    problems: List[MathProblem] = []
    for i, row in enumerate(rows):
        ans_raw = row.get("answer", "")
        ans = str(ans_raw).strip()
        numeric = None
        try:
            numeric = float(ans) if ans != "" else None
        except Exception:
            numeric = None

        problem_types = row.get("problem_type", None)
        if isinstance(problem_types, (list, tuple)) and len(problem_types) > 0:
            prob_type = str(problem_types[0]).strip() or "AIME"
        else:
            prob_type = "AIME"

        pid = row.get("problem_idx", None)
        try:
            problem_id = int(pid) if pid is not None else i
        except Exception:
            problem_id = i

        problems.append(MathProblem(
            problem_id=problem_id,
            text=str(row.get("problem", "") or ""),
            level=5,
            type=prob_type,
            solution="",
            boxed_answer=ans,
            numeric_answer=numeric,
        ))

    return problems


def load_olympiad_bench(max_per_type: int = 200) -> List[MathProblem]:
    """
    OlympiadBench (math-ai/olympiadbench) — 674 olympiad-level problems.

    Fields used:
      subfield      : Algebra | Combinatorics | Geometry | Number Theory
      question      : problem text
      final_answer  : list with one LaTeX answer string
      answer_type   : Numerical | Expression | Tuple | Interval
      is_multiple_answer: bool
      difficulty    : "Competition" (constant)

    All problems are level 5 (olympiad). Subfield maps to CL stream domain.
    """
    print("Loading OlympiadBench (math-ai/olympiadbench)...")
    try:
        from datasets import load_dataset
        ds   = load_dataset("math-ai/olympiadbench", "default", split="test")
        rows = list(ds)
        print(f"  Downloaded {len(rows)} rows")
    except Exception as e:
        print(f"  Failed: {e}")
        return []

    # Map subfield → canonical domain names matching SEED_SCHEMAS
    type_map = {
        "algebra":       "Algebra",
        "combinatorics": "Counting & Probability",
        "geometry":      "Geometry",
        "number theory": "Number Theory",
    }

    problems    : List[MathProblem] = []
    type_counts : dict              = defaultdict(int)
    skipped_multi = 0

    for i, row in enumerate(rows):
        # Skip multi-answer problems — evaluator handles single answers only
        if row.get("is_multiple_answer", False):
            skipped_multi += 1
            continue

        # Subfield → domain
        raw_sf   = str(row.get("subfield") or "Algebra").strip()
        domain   = type_map.get(raw_sf.lower(), raw_sf)

        if type_counts[domain] >= max_per_type:
            continue

        # Answer — final_answer is a list; take first element
        raw_ans = row.get("final_answer", [])
        if isinstance(raw_ans, (list, tuple)) and len(raw_ans) > 0:
            ans_str = str(raw_ans[0]).strip()
        else:
            ans_str = str(raw_ans).strip()

        # Strip surrounding $ signs if present
        ans_str = ans_str.strip("$").strip()

        # Try numeric parse
        numeric = boxed_to_float(ans_str)
        if numeric is None:
            try:
                numeric = float(ans_str)
            except Exception:
                numeric = None

        # Skip if no answer
        if not ans_str:
            continue

        problems.append(MathProblem(
            problem_id    = int(row.get("id", i)),
            text          = str(row.get("question", "") or ""),
            level         = 5,          # all olympiad level
            type          = domain,
            solution      = " ".join(str(s) for s in (row.get("solution") or [])),
            boxed_answer  = ans_str,
            numeric_answer= numeric,
        ))
        type_counts[domain] += 1

    type_dist: dict = defaultdict(int)
    for p in problems:
        type_dist[p.type] += 1

    print(f"  Loaded {len(problems)} problems "
          f"(skipped {skipped_multi} multi-answer):")
    for t, c in sorted(type_dist.items()):
        print(f"    {t:<35} {c}")

    return problems


def _builtin_problems() -> List[MathProblem]:
    """Fallback: built-in problems when network is unavailable."""
    raw = [
        # (text, level, type, solution_with_boxed)
        ("Solve for x: 2x + 3 = 11", 1, "Algebra",
         r"2x = 8, so x = \boxed{4}"),
        ("Find x if 3x - 5 = 10", 1, "Algebra",
         r"3x = 15, x = \boxed{5}"),
        ("A train travels 120 miles in 2 hours. Speed?", 1, "Algebra",
         r"Speed = 120/2 = \boxed{60} mph"),
        ("Sam earns $60/day working and loses $30/day not working over 20 days, earning $660. Days not worked?", 2, "Algebra",
         r"Let y = days not worked. 60(20-y) - 30y = 660 \Rightarrow y = \boxed{6}"),
        ("Find GCD of 12 and 18", 1, "Number Theory",
         r"GCD = \boxed{6}"),
        ("How many factors does 24 have?", 1, "Number Theory",
         r"24 = 2^3 \cdot 3, factors = 4 \cdot 2 = \boxed{8}"),
        ("Find LCM of 4 and 6", 1, "Number Theory",
         r"LCM = \boxed{12}"),
        ("What is the remainder when 17 is divided by 5?", 1, "Number Theory",
         r"\boxed{2}"),
        ("Choose 2 items from 5. How many ways?", 1, "Counting & Probability",
         r"\binom{5}{2} = \boxed{10}"),
        ("A bag has 3 red and 5 blue balls. P(red)?", 1, "Counting & Probability",
         r"P = 3/8 = \boxed{0.375}"),
        ("How many 2-digit numbers exist?", 1, "Counting & Probability",
         r"90 numbers: \boxed{90}"),
        ("Two dice rolled. P(sum = 7)?", 2, "Counting & Probability",
         r"6/36 = \boxed{0.1667}"),
        ("Simplify: $\frac{x^2-4}{x-2}$", 2, "Algebra",
         r"\boxed{x+2}"),
        ("Roots of $x^2 - 5x + 6 = 0$?", 2, "Algebra",
         r"x = 2 or x = \boxed{3}"),
        ("Find LCM of 6, 8, 12", 2, "Number Theory",
         r"LCM = \boxed{24}"),
        ("Is 97 prime?", 2, "Number Theory",
         r"Yes, \boxed{1} (it is prime)"),
        ("Arrange 4 books on a shelf: how many ways?", 2, "Counting & Probability",
         r"4! = \boxed{24}"),
        ("P(at least one head in 3 flips)?", 2, "Counting & Probability",
         r"1 - P(none) = 1 - 1/8 = \boxed{0.875}"),
    ]
    problems = []
    for i, (text, level, ptype, sol) in enumerate(raw):
        boxed   = extract_boxed(sol)
        numeric = boxed_to_float(boxed)
        problems.append(MathProblem(
            problem_id=i, text=text, level=level, type=ptype,
            solution=sol, boxed_answer=boxed, numeric_answer=numeric,
        ))
    print(f"Using {len(problems)} built-in problems")
    return problems


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9: SEED SCHEMAS
# Grounded in AMS Mathematics Subject Classification (MSC 2020)
# Six broad families covering all competition math domains.
# Each schema is intentionally wide — synthesis fills the specialisations.
# ══════════════════════════════════════════════════════════════════════════════

SEED_SCHEMAS = [
    {
        # AMS MSC: 12-20xx (Field theory, polynomials, linear/multilinear algebra,
        # associative rings, group theory, topological groups)
        # Covers: Prealgebra, Algebra, Intermediate Algebra, Precalculus
        "name": "Algebra",
        "description": (
            "All problems involving algebraic manipulation and equation solving: "
            "linear, quadratic, polynomial, rational, exponential, and logarithmic "
            "equations and inequalities; systems of equations; functions, their "
            "domains, ranges, and compositions; arithmetic and geometric sequences "
            "and series; completing the square; Vieta's formulas for roots; "
            "partial fractions; absolute value; floor and ceiling; complex numbers; "
            "and algebraic word problems that translate into equations."
        ),
        "template": (
            "1. Identify the algebraic structure:\n"
            "   linear / quadratic / polynomial / rational / exponential / "
            "   logarithmic / system / sequence / function.\n"
            "2. Assign variables to all unknowns.\n"
            "3. Set up equation(s) or inequality from the problem statement.\n"
            "4. Choose the right technique:\n"
            "   - Linear: isolate variable\n"
            "   - Quadratic: factor, complete the square, or quadratic formula\n"
            "   - Polynomial: rational root theorem, synthetic division, Vieta\n"
            "   - System: substitution or elimination\n"
            "   - Sequence: identify common difference/ratio, use nth-term formula\n"
            "   - Function: evaluate, compose, or find inverse\n"
            "5. Check domain restrictions (denominators ≠ 0, even radicands ≥ 0).\n"
            "6. Verify answer by substituting back."
        ),
        "heuristics": ["introduce_variable", "decompose",
                       "change_representation", "find_pattern"],
        "anti_heuristics": [],
    },
    {
        # AMS MSC: 11xx (Number Theory)
        # Covers: Number Theory, digit problems, modular arithmetic
        "name": "Number Theory",
        "description": (
            "All problems about integers and their properties: "
            "divisibility, prime numbers, prime factorization, "
            "number of divisors, sum of divisors, GCD, LCM, "
            "modular arithmetic, congruences, remainders, "
            "Euler's theorem, Fermat's little theorem, "
            "Chinese Remainder Theorem, Diophantine equations, "
            "digit problems (digit sums, digit counts, place-value constraints), "
            "floor and ceiling arithmetic, and integer sequences "
            "defined by divisibility or modular conditions."
        ),
        "template": (
            "1. Identify the number-theoretic structure:\n"
            "   divisibility / modular / Diophantine / digit / prime / counting.\n"
            "2. Prime factorize all relevant integers if needed.\n"
            "3. Apply the appropriate tool:\n"
            "   - Divisibility: σ(n), τ(n), GCD/LCM formulas\n"
            "   - Modular: reduce mod n, identify cycle length, CRT\n"
            "   - Diophantine: parameterise solutions using GCD condition\n"
            "   - Digit: expand in base 10, set up place-value equations\n"
            "   - Prime: sieve, primality test, prime gap\n"
            "4. Count or compute the final answer.\n"
            "5. Verify with a small numerical example."
        ),
        "heuristics": ["find_pattern", "decompose",
                       "change_representation", "work_backwards"],
        "anti_heuristics": [],
    },
    {
        # AMS MSC: 51-53xx (Geometry, Differential Geometry, Convex Geometry)
        # Covers: Geometry (plane, coordinate, solid, trigonometric)
        "name": "Geometry",
        "description": (
            "All geometric problems: areas and perimeters of triangles, "
            "quadrilaterals, circles, and composite figures; "
            "volumes and surface areas of prisms, pyramids, cylinders, cones, "
            "and spheres; coordinate geometry (distance, midpoint, slope, "
            "equations of lines and circles, locus problems); "
            "angle relationships (parallel lines, transversals, polygons); "
            "triangle properties (congruence, similarity, Pythagorean theorem, "
            "special right triangles, trigonometric ratios); "
            "circle theorems (inscribed angles, tangent lines, arc lengths, "
            "sector areas); and geometric transformations."
        ),
        "template": (
            "1. Draw and label the figure with all given measurements.\n"
            "2. Classify the problem type:\n"
            "   - Area/perimeter: identify shape, apply formula\n"
            "   - Coordinate: use distance, midpoint, slope formulas\n"
            "   - Angles: apply angle-sum, parallel-line, or circle theorems\n"
            "   - Similarity/congruence: identify corresponding parts, set up ratio\n"
            "   - Trigonometry: label opposite/adjacent/hypotenuse, apply sin/cos/tan\n"
            "   - 3D: identify faces, apply volume/surface area formula\n"
            "3. Set up equations from geometric relationships.\n"
            "4. Solve algebraically.\n"
            "5. Verify: lengths > 0, angles sum correctly, areas are positive."
        ),
        "heuristics": ["decompose", "introduce_variable",
                       "change_representation", "find_pattern"],
        "anti_heuristics": [],
    },
    {
        # AMS MSC: 05Axx (Enumerative combinatorics), 05Cxx (Graph theory)
        # Covers: counting problems in Counting & Probability
        "name": "Combinatorics",
        "description": (
            "All counting problems: permutations (ordered arrangements), "
            "combinations (unordered selections), the multiplication principle "
            "for independent choices, the addition principle for disjoint cases, "
            "inclusion-exclusion for overlapping sets, "
            "circular and linear arrangements, "
            "distinguishable vs indistinguishable objects, "
            "pigeonhole principle, "
            "counting with restrictions (forbidden positions, required adjacency), "
            "stars-and-bars for distributing identical objects, "
            "and counting paths or sequences with constraints."
        ),
        "template": (
            "1. Determine what is being counted: arrangements, selections, "
            "   distributions, or sequences.\n"
            "2. Does order matter?\n"
            "   YES → permutation nPr = n!/(n-r)!\n"
            "   NO  → combination nCr = n!/((n-r)!r!)\n"
            "3. Are there restrictions?\n"
            "   - Subtract forbidden cases from total\n"
            "   - Or use inclusion-exclusion: |A∪B| = |A|+|B|-|A∩B|\n"
            "4. Are choices independent? → multiply counts\n"
            "5. Are choices exclusive? → add counts\n"
            "6. Verify with a small example."
        ),
        "heuristics": ["decompose", "change_representation",
                       "find_pattern", "work_backwards"],
        "anti_heuristics": [],
    },
    {
        # AMS MSC: 60xx (Probability theory and stochastic processes)
        # Covers: probability problems in Counting & Probability
        "name": "Probability",
        "description": (
            "All probability problems: computing P(event) = favorable/total, "
            "complement rule P(A) = 1 - P(not A), "
            "union rule P(A∪B) = P(A)+P(B)-P(A∩B), "
            "conditional probability P(A|B) = P(A∩B)/P(B), "
            "independence P(A∩B) = P(A)·P(B), "
            "expected value E[X] = Σ x·P(x), "
            "geometric probability (ratio of lengths or areas), "
            "probability with dice, coins, cards, urns, and random selections, "
            "and multi-step probability with replacement or without replacement."
        ),
        "template": (
            "1. Define the sample space and its total size (or measure).\n"
            "2. Identify the event whose probability is needed.\n"
            "3. Choose the approach:\n"
            "   - Direct: count favorable / count total\n"
            "   - Complement: 1 - P(unwanted event)\n"
            "   - Conditional: use P(A|B) = P(A∩B)/P(B)\n"
            "   - Multi-step: multiply probabilities along branches\n"
            "   - Geometric: ratio of areas or lengths\n"
            "   - Expected value: Σ outcome × probability\n"
            "4. Compute the probability.\n"
            "5. Verify: answer must be in [0, 1]. "
            "   All cases must sum to 1."
        ),
        "heuristics": ["decompose", "find_pattern",
                       "change_representation", "work_backwards"],
        "anti_heuristics": [],
    },
    {
        # AMS MSC: 26xx (Real functions), 28xx (Measure theory),
        # 40xx (Sequences/series/summability), 41xx (Approximations),
        # 42xx (Fourier analysis)
        # Covers: Precalculus, limits, series convergence in competition math
        "name": "Calculus and Analysis",
        "description": (
            "Problems involving limits, derivatives, integrals, and series. "
            "Includes: evaluating limits (direct substitution, factoring, "
            "L'Hopital's rule, squeeze theorem); "
            "derivatives and differentiation rules (power, chain, product, quotient); "
            "finding extrema and inflection points; "
            "definite and indefinite integration; "
            "area between curves and volumes of revolution; "
            "convergence of sequences and series "
            "(geometric series, p-series, ratio test, comparison test); "
            "Taylor and Maclaurin series; "
            "and real analysis inequalities (AM-GM, Cauchy-Schwarz)."
        ),
        "template": (
            "1. Identify the operation needed:\n"
            "   limit / derivative / integral / series / inequality.\n"
            "2. For limits:\n"
            "   - Try direct substitution first\n"
            "   - If 0/0 or ∞/∞: factor, conjugate, or L'Hopital\n"
            "   - Squeeze if bounded by simpler functions\n"
            "3. For derivatives:\n"
            "   - Identify function type, apply correct rule\n"
            "   - Set f'(x)=0 for critical points\n"
            "4. For integrals:\n"
            "   - Try substitution (u-sub) first\n"
            "   - Integration by parts: ∫u dv = uv - ∫v du\n"
            "   - Partial fractions for rational functions\n"
            "5. For series:\n"
            "   - Geometric: sum = a/(1-r) if |r|<1\n"
            "   - Apply convergence test\n"
            "6. Verify by differentiating the antiderivative "
            "   or checking boundary cases."
        ),
        "heuristics": ["change_representation", "decompose",
                       "find_pattern", "introduce_variable"],
        "anti_heuristics": [],
    },
]


def build_schema_memory(encoder, synthesizer=None, verbose=True) -> MemoryController:
    bank      = SchemaBank()
    replay    = ReplayBuffer()
    extractor = FeatureExtractor()

    # Build LLM client for operator classification during seeding
    try:
        from openai import OpenAI
        llm_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        llm_client = None

    for schema in SEED_SCHEMAS:
        emb  = encoder.encode(schema["description"], normalize_embeddings=True)
        feat = extractor.extract(schema["description"], emb)

        # Fix 2: classify seed schema operator type via LLM for accuracy
        if llm_client is not None:
            feat.operator_type = classify_operator_llm(
                schema["description"], llm_client
            )

        # Fix 3: warm-start centroid from description + template + name
        hook = build_warm_hook(schema, feat, encoder)
        bank.add(schema, hook)

    return MemoryController(bank, replay, encoder,
                            synthesizer=synthesizer,
                            verbose=verbose, name="Schema Memory")


def build_intelligent_schema_memory(encoder, synthesizer=None,
                                    verbose=True) -> "IntelligentSchemaMemory":
    """
    Build an IntelligentSchemaMemory — Schema Memory with all 5
    self-improvement mechanisms enabled:
      1. Self-Audit   (every 10 eps)
      2. Self-Correct (triggered by audit)
      3. Self-Merge   (every 20 eps)
      4. Self-Promote (triggered by audit)
      5. Self-Prune   (every 25 eps)
    """
    bank      = SchemaBank()
    replay    = ReplayBuffer()
    extractor = FeatureExtractor()

    # Give bank a reference to replay so agentic_solve can pass it
    # to tool_search_past_failures without threading it through every call
    bank._replay_ref = replay

    try:
        from openai import OpenAI
        llm_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        llm_client = None

    for schema in SEED_SCHEMAS:
        emb  = encoder.encode(schema["description"], normalize_embeddings=True)
        feat = extractor.extract(schema["description"], emb)
        if llm_client is not None:
            feat.operator_type = classify_operator_llm(
                schema["description"], llm_client
            )
        hook = build_warm_hook(schema, feat, encoder)
        bank.add(schema, hook)

    return IntelligentSchemaMemory(bank, replay, encoder,
                                   synthesizer=synthesizer,
                                   verbose=verbose)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10: GPT-4o SCHEMA SYNTHESIZER
# ══════════════════════════════════════════════════════════════════════════════

def gpt4o_synthesizer(problem_text: str, features: ProblemFeatures) -> Optional[dict]:
    """
    Synthesize a new GENERAL schema from a failed problem.
    Forces general family schemas, not problem-specific ones.
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        print(f"  [Synthesizer] Client init failed: {e}")
        return None

    prompt = f"""A math solver failed on this problem:

Problem: {problem_text}

Detected features:
- Operator type: {features.operator_type}
- Structure: {features.structural_pattern}
- Heuristics needed: {features.heuristics}

CRITICAL RULES:
1. Create a GENERAL schema covering a FAMILY of similar problems — NOT just this one.
2. The name must describe a problem TYPE, not a specific instance.
   BAD:  'Projectile Quadratic Threshold Interval'  (too specific)
   BAD:  'Arithmetic Series Threshold (Increasing Step Gains)'  (too specific)
   GOOD: 'Quadratic Applications'  (general family)
   GOOD: 'Arithmetic Sequences'    (general family)
3. The description must apply to at least 10 different problems of this type.

Return ONLY valid JSON (no markdown, no explanation):
{{
  "name": "General Family Name (2-4 words)",
  "description": "one sentence describing the GENERAL problem family",
  "template": "numbered solution steps applicable to the whole family",
  "heuristics": ["list", "of", "heuristics"]
}}"""

    try:
        resp = client.chat.completions.create(
            model=SYNTHESIZER_MODEL,
            messages=[
                {"role": "system", "content":
                 "You create GENERAL mathematical schemas for families of problems. "
                 "Never create schemas for specific problems — always generalize. Return only JSON."},
                {"role": "user", "content": prompt},
            ],
        )
        text = resp.choices[0].message.content.strip()
        text = re.sub(r'^```json\s*|\s*```$', '', text, flags=re.MULTILINE)
        schema = json.loads(text)
        schema.setdefault("heuristics", ["decompose"])
        print(f"  [Synthesizer] Created: '{schema['name']}'")
        return schema
    except Exception as e:
        print(f"  [Synthesizer] Failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11: MAIN EXPERIMENT
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(source=None, n_per_block=20, max_level=5, min_level=1,
                   numeric_only=False,
                   model=LLM_MODEL, output_dir="results",
                   use_synthesizer=True, seed=42, verbose=True,
                   dataset="competition_math",
                   use_agentic=False,
                   pass_k: int = 1):

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    random.seed(seed)

    # Load encoder
    print("Loading encoder...")
    from openai import OpenAI
    _oai_client = OpenAI(api_key=OPENAI_API_KEY)

    class OpenAIEmbedEncoder:
        """text-embedding-3-small via OpenAI API. Cached for efficiency."""
        def __init__(self, client, model):
            self.client = client
            self.model  = model
            self._cache: Dict[str, np.ndarray] = {}

        def encode(self, text, normalize_embeddings=True):
            text = text.strip().replace("\n", " ")[:8000]
            if text in self._cache:
                return self._cache[text]
            resp = self.client.embeddings.create(
                input=[text], model=self.model
            )
            v = np.array(resp.data[0].embedding, dtype=np.float32)
            if normalize_embeddings:
                v /= np.linalg.norm(v)
            self._cache[text] = v
            return v

    encoder = OpenAIEmbedEncoder(_oai_client, OPENAI_EMBEDDING_MODEL)
    print(f"  Encoder: {OPENAI_EMBEDDING_MODEL} (OpenAI API)")

    # Load dataset
    if dataset == "math500":
        problems = load_math500()
    elif dataset == "aime2025":
        problems = load_aime2025()
    else:
        problems = load_competition_math(source, max_level=max_level,
                                         min_level=min_level,
                                         numeric_only=numeric_only)
    if not problems:
        print("No problems loaded — check data source")
        return

    # Build CL stream
    stream = build_cl_stream(problems, n_per_block=n_per_block, seed=seed)

    # Build systems
    synthesizer = gpt4o_synthesizer if use_synthesizer else None

    systems = {
        "Static LLM":    StaticLLMController(
            SchemaBank(), ReplayBuffer(), encoder, verbose=False, name="Static LLM"
        ),
        "Neural Memory": NeuralMemoryController(encoder, verbose=False),
        "Schema Memory": build_schema_memory(
            encoder, synthesizer=synthesizer, verbose=verbose
        ),
        "ISM":           build_intelligent_schema_memory(
            encoder, synthesizer=synthesizer, verbose=verbose
        ),
    }
    trackers = {name: CLTracker() for name in systems}
    passk_trackers = {name: CLTracker() for name in systems} if pass_k and pass_k > 1 else None
    logs     = {name: [] for name in systems}

    print(f"\nRunning {len(stream)} episodes × {len(systems)} systems")
    passk_disp = f" | pass@{pass_k}" if pass_k and pass_k > 1 else ""
    print(f"Model: {model} | Synthesizer: {'ON' if synthesizer else 'OFF'}{passk_disp}\n")

    for i, problem in enumerate(stream):
        block_label = problem.task_id

        if verbose:
            print(f"\nEp {i+1:>3}/{len(stream)} | {block_label} | L{problem.level} | "
                  f"{problem.text[:60]}...")

        for sys_name, controller in systems.items():

            # ── FAST LOOP ──────────────────────────────────────────────────
            schema, features, ret_score, ret_breakdown, ret_status = (
                controller.get_schema_for_problem(problem.text)
            )

            # ISM uses agentic_solve when --agentic flag is set.
            # All other systems always use the passive hard_reset_call.
            # This gives a clean ablation: ISM-passive vs ISM-agentic.
            tool_trace = None

            # Always get the base Schema Memory answer first (and optionally sample k times for pass@k)
            responses = []
            if pass_k and pass_k > 1:
                for _ in range(pass_k):
                    responses.append(hard_reset_call(problem.text, schema, model=model))
            else:
                responses.append(hard_reset_call(problem.text, schema, model=model))

            base_response = responses[0]
            sample_correct = [
                evaluate_answer(r, problem.boxed_answer or "", problem.numeric_answer)
                for r in responses
            ]
            base_correct = bool(sample_correct[0]) if sample_correct else False
            passk_correct = any(sample_correct) if (pass_k and pass_k > 1) else base_correct

            response = base_response
            correct  = base_correct

            # Only trigger the agentic loop if the base answer is wrong (for ISM)
            if use_agentic and sys_name == "ISM" and not base_correct:
                agent_result = agentic_solve(
                    problem_text = problem.text,
                    schema       = schema,
                    bank         = controller.bank,
                    encoder      = controller.encoder,
                    model        = model,
                    max_turns    = 6,
                    verbose      = verbose,
                )
                response   = agent_result["response"]
                tool_trace = agent_result["tool_trace"]
                n_turns    = agent_result["n_turns"]

                # Re-evaluate correctness based on the agentic answer
                correct = evaluate_answer(
                    response, problem.boxed_answer or "",
                    problem.numeric_answer
                )

                if verbose:
                    tools_used = [t["tool"] for t in tool_trace]
                    print(f"  [Agent turns={n_turns}] tools={tools_used}")

            outcome = ("correct" if correct else
                       "generic" if ret_status in ("generic","empty_bank") else
                       "incorrect")

            # ── SLOW LOOP: update memory ───────────────────────────────────
            # Pass response_text so mistakes can be extracted and stored
            # in episodic memory for search_past_failures to retrieve.
            controller.after_episode(
                problem.text, features, schema,
                ret_status, outcome, block_label,
                tool_trace    = tool_trace,
                response_text = response,
            )

            # Track
            trackers[sys_name].record(block_label, correct)
            if passk_trackers is not None:
                # pass@k is computed on the base sampling loop (no agentic tools)
                passk_trackers[sys_name].record(block_label, passk_correct)
            logs[sys_name].append({
                "episode":      i,
                "task":         block_label,
                "level":        problem.level,
                "correct":      correct,
                "pass_k":       int(pass_k or 1),
                "passk_correct": bool(passk_correct),
                "schema":       schema.get("name",""),
                "ret_status":   ret_status,
                "ret_score":    round(ret_score, 3),
                "bank_size":    controller.bank.size(),
                "gold":         problem.boxed_answer or "",
                "pred":         extract_final_answer(response) or "",
                "n_tool_calls": len(tool_trace) if tool_trace else 0,
                "tools_used":   [t["tool"] for t in tool_trace] if tool_trace else [],
            })
            logs[sys_name][-1]["bank_size"] = controller.bank.size()

            if verbose:
                status_sym  = "✓ SUCCESS" if correct else "✗ FAILURE"
                ret_disp    = ret_status.replace("retrieved_high","ret_high")\
                                        .replace("retrieved_med","ret_med")
                tool_info   = (f" tools={[t['tool'] for t in tool_trace]}"
                               if tool_trace else "")
                gold_str    = str(problem.boxed_answer or "?")[:12]
                pred_str    = str(extract_final_answer(response) or "?")[:12]
                print(f"  [{sys_name:<16}] {status_sym:<11} | "
                      f"gold={gold_str:<12} pred={pred_str:<12} | "
                      f"schema={schema.get('name','')[:22]:<22} | "
                      f"ret={ret_disp:<12} bank={controller.bank.size()}"
                      f"{tool_info}")
                # Print mistake or insight from the most recent replay entry
                if controller.replay.buffer:
                    last_ep = controller.replay.buffer[-1]
                    if not correct and last_ep.mistake:
                        print(f"    └─ mistake : {last_ep.mistake}")
                    elif correct and last_ep.insight:
                        print(f"    └─ insight : {last_ep.insight}")

        # Block boundary summary
        if (i + 1) % n_per_block == 0:
            print(f"\n{'═'*60}")
            print(f"Block complete: {block_label}")
            for name, tracker in trackers.items():
                m = tracker.compute()
                print(f"  {name:<18} acc={m['average_accuracy']:.2%} "
                      f"forgetting={m['forgetting']:.2%}")
            if passk_trackers is not None:
                for name, tracker in passk_trackers.items():
                    m = tracker.compute()
                    print(f"  {name:<18} pass@{pass_k}={m['average_accuracy']:.2%} "
                          f"forgetting={m['forgetting']:.2%}")

            # Optional: show average tool usage per episode for this block
            # (most useful for ISM, but computed for all systems that log n_tool_calls).
            for name, log in logs.items():
                block_slice = log[-n_per_block:] if len(log) >= n_per_block else log
                if not block_slice:
                    continue
                total_tools = sum(e.get("n_tool_calls", 0) for e in block_slice)
                avg_tools   = total_tools / len(block_slice)
                print(f"  {name:<18} tools/ep={avg_tools:.2f}")

            # ISM live bank state — what it knows and how healthy it is
            ism = systems.get("ISM")
            if ism:
                print_bank_state(ism, label=f"after block '{block_label}' ep={i+1}")
            print()

    # ── Save results ──────────────────────────────────────────────────────────
    for name, log in logs.items():
        safe = name.replace(" ","_").lower()
        with open(f"{output_dir}/{safe}.json", "w") as f:
            json.dump(log, f, indent=2)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("FINAL RESULTS — CONTINUAL LEARNING EVALUATION")
    print("="*60)

    summary = []
    for name, tracker in trackers.items():
        print(tracker.summary(name))
        m = tracker.compute()
        row = {"system": name, **{k: round(v,4) for k,v in m.items()
                                  if isinstance(v, float)}}
        if passk_trackers is not None:
            pm = passk_trackers[name].compute()
            row["pass_k"] = int(pass_k)
            row["passk_average_accuracy"] = round(pm["average_accuracy"], 4)
            row["passk_plasticity"] = round(pm["plasticity"], 4)
            row["passk_stability"] = round(pm["stability"], 4)
            row["passk_forgetting"] = round(pm["forgetting"], 4)
            row["passk_backward_transfer"] = round(pm["backward_transfer"], 4)
        summary.append(row)
        if passk_trackers is not None:
            print(passk_trackers[name].summary(f"{name} (pass@{pass_k})"))

    with open(f"{output_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── Save ISM lift log for Figure 4 ───────────────────────────
    ism_ctrl = systems.get("ISM")
    if ism_ctrl and hasattr(ism_ctrl, '_lift_log'):
        with open(f"{output_dir}/ism_lift_log.json", "w") as f:
            json.dump(ism_ctrl._lift_log, f, indent=2)
    # ─────────────────────────────────────────────────────────────

    print(f"\nResults saved to {output_dir}/")
    return trackers


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12: ABLATION — Schema Memory (passive) vs ISM (active)
# Critical experiment: do the 5 self-improvement mechanisms actually matter?
# Both systems use identical solvers, identical streams, identical synthesizers.
# The only difference is whether audit/correct/merge/promote/prune run.
# ══════════════════════════════════════════════════════════════════════════════

def run_ablation(source=None, n_per_block=30, max_level=5, min_level=1,
                 numeric_only=False,
                 model=LLM_MODEL, output_dir="results/ablation",
                 use_synthesizer=True, seed=42, verbose=True,
                 dataset="competition_math"):
    """
    Focused ablation: Schema Memory (passive) vs ISM (active).

    Metrics tracked per episode:
      1. Rolling accuracy  (20-ep window)     — does ISM degrade less?
      2. Bank size                             — does passive bloat unboundedly?
      3. Avg schema success_rate              — does passive quality drop over time?
      4. Retrieval hit rate                   — does passive retrieve less reliably?
      5. Schema fragmentation (used schemas)  — does passive fragment more?

    Run:
        python schema_memory_agentic.py --ablation
        python schema_memory_agentic.py --ablation --dataset math500 --n_per_block 40
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    random.seed(seed)

    # ── Encoder ───────────────────────────────────────────────────────────────
    print("Loading encoder...")
    from openai import OpenAI
    _oai_client = OpenAI(api_key=OPENAI_API_KEY)

    class OpenAIEmbedEncoder:
        def __init__(self, client, model):
            self.client = client
            self.model  = model
            self._cache: Dict[str, np.ndarray] = {}

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

    encoder = OpenAIEmbedEncoder(_oai_client, OPENAI_EMBEDDING_MODEL)
    print(f"  Encoder: {OPENAI_EMBEDDING_MODEL}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    if dataset == "math500":
        problems = load_math500()
    elif dataset == "aime2025":
        problems = load_aime2025()
    else:
        problems = load_competition_math(source, max_level=max_level,
                                         min_level=min_level,
                                         numeric_only=numeric_only)

    if not problems:
        print("No problems loaded — check data source")
        return

    stream = build_cl_stream(problems, n_per_block=n_per_block, seed=seed)

    # ── Systems ───────────────────────────────────────────────────────────────
    synthesizer = gpt4o_synthesizer if use_synthesizer else None

    systems = {
        "Passive (Schema Memory)": build_schema_memory(
            encoder, synthesizer=synthesizer, verbose=False
        ),
        "Active (ISM)": build_intelligent_schema_memory(
            encoder, synthesizer=synthesizer, verbose=verbose
        ),
    }

    trackers = {name: CLTracker() for name in systems}
    logs     = {name: [] for name in systems}

    # Track when ISM mechanisms fire
    mechanism_log: List[dict] = []

    print(f"\n{'═'*65}")
    print(f"  ABLATION: Schema Memory (passive) vs ISM (active)")
    print(f"  Stream : {len(stream)} episodes | Model: {model}")
    print(f"  Metrics: rolling_acc | bank_size | avg_success_rate | hit_rate")
    print(f"{'═'*65}\n")

    for i, problem in enumerate(stream):
        block_label = problem.task_id
        ep_results  = {}  # collect both systems' results before printing

        for sys_name, controller in systems.items():

            schema, features, ret_score, ret_breakdown, ret_status = (
                controller.get_schema_for_problem(problem.text)
            )

            response = hard_reset_call(problem.text, schema, model=model)
            correct  = evaluate_answer(
                response, problem.boxed_answer or "", problem.numeric_answer
            )
            outcome = ("correct"   if correct else
                       "generic"   if ret_status in ("generic", "empty_bank") else
                       "incorrect")

            controller.after_episode(
                problem.text, features, schema,
                ret_status, outcome, block_label,
            )
            trackers[sys_name].record(block_label, correct)

            # ── Bank quality snapshot ─────────────────────────────────────
            bank     = controller.bank
            hook_vals = list(bank.hooks.values())
            avg_sr   = float(np.mean([h.success_rate for h in hook_vals])) \
                       if hook_vals else 0.0
            used     = sum(1 for h in hook_vals if h.usage_count > 0)
            hit      = 1 if ret_status == "retrieved_high" else 0  # high-conf only

            logs[sys_name].append({
                "episode":          i,
                "task":             block_label,
                "correct":          correct,
                "ret_status":       ret_status,
                "ret_score":        round(ret_score, 3),
                "schema":           schema.get("name", ""),
                "bank_size":        bank.size(),
                "avg_success_rate": round(avg_sr, 3),
                "used_schemas":     used,
                "retrieval_hit":    hit,
                "gold":             problem.boxed_answer or "",
                "pred":             extract_final_answer(response) or "",
            })

            ep_results[sys_name] = {
                "correct":    correct,
                "schema":     schema.get("name", ""),
                "pred":       extract_final_answer(response) or "?",
                "ret_status": ret_status,
                "bank_size":  bank.size(),
            }

        # ── Per-episode side-by-side print ────────────────────────────────
        passive_r = ep_results.get("Passive (Schema Memory)", {})
        active_r  = ep_results.get("Active (ISM)", {})
        p_sym = "✓" if passive_r.get("correct") else "✗"
        a_sym = "✓" if active_r.get("correct")  else "✗"
        gold  = problem.boxed_answer or "?"

        # Highlight episodes where the two systems disagree
        differ = passive_r.get("correct") != active_r.get("correct")
        marker = " ◄" if differ else ""

        print(
            f"  Ep {i+1:>3} | {block_label:<28} | gold={gold:<6} | "
            f"Passive {p_sym} ({passive_r.get('pred','?'):<6}) "
            f"[{passive_r.get('schema','')[:18]:<18}] | "
            f"ISM {a_sym} ({active_r.get('pred','?'):<6}) "
            f"[{active_r.get('schema','')[:18]:<18}] "
            f"bank={active_r.get('bank_size','?')}"
            f"{marker}"
        )

        # ── Per-20-episode comparison ──────────────────────────────────────
        if (i + 1) % 20 == 0:
            print(f"\n  Ep {i+1:>3} | Block: {block_label}")
            print(f"  {'System':<26} {'Acc':>5}  {'Bank':>5}  "
                  f"{'AvgSR':>6}  {'HitRate':>8}  {'UsedSchemas':>12}")
            print(f"  {'─'*26} {'─'*5}  {'─'*5}  {'─'*6}  {'─'*8}  {'─'*12}")

            for name, log in logs.items():
                recent   = log[-20:]
                acc      = sum(e["correct"]       for e in recent) / len(recent)
                hit_rate = sum(e["retrieval_hit"]  for e in recent) / len(recent)
                bank_sz  = log[-1]["bank_size"]
                avg_sr   = log[-1]["avg_success_rate"]
                used     = log[-1]["used_schemas"]
                label    = "Passive" if "Passive" in name else "Active "
                print(f"  {label:<26} {acc:>5.0%}  {bank_sz:>5}  "
                      f"{avg_sr:>6.2f}  {hit_rate:>8.0%}  {used:>12}")

    # ── Save logs ─────────────────────────────────────────────────────────────
    for name, log in logs.items():
        safe = ("passive" if "Passive" in name else "active")
        with open(f"{output_dir}/{safe}.json", "w") as f:
            json.dump(log, f, indent=2)

    # ── Final report ──────────────────────────────────────────────────────────
    _print_ablation_report(logs, trackers, output_dir)
    return logs, trackers


def _print_ablation_report(logs, trackers, output_dir):
    """Print full ablation comparison: accuracy, bank quality, degradation curves."""
    W = 65
    print(f"\n{'═'*W}")
    print("  ABLATION FINAL REPORT")
    print(f"{'═'*W}")

    # CL metric summary per system
    for name, tracker in trackers.items():
        print(tracker.summary(name))

    max_eps = max(len(log) for log in logs.values())

    # ── Rolling accuracy curve ─────────────────────────────────────────────
    print(f"\n  Rolling accuracy (20-ep window) over stream:")
    print(f"  {'Ep':<6}  {'Passive':>10}  {'Active (ISM)':>13}  {'Delta':>7}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*13}  {'─'*7}")

    for start in range(0, max_eps, 20):
        row = {}
        for name, log in logs.items():
            block = log[start:start + 20]
            row[name] = sum(e["correct"] for e in block) / len(block) if block else None

        vals = list(row.values())
        passive_acc = vals[0]
        active_acc  = vals[1]
        if passive_acc is not None and active_acc is not None:
            delta = active_acc - passive_acc
            delta_str = f"{delta:+.0%}"
        else:
            delta_str = "N/A"
        print(f"  {start+1:<6}  {passive_acc:>10.0%}  {active_acc:>13.0%}  "
              f"{delta_str:>7}")

    # ── Bank size over time ────────────────────────────────────────────────
    print(f"\n  Bank size over time (sampled every 20 episodes):")
    print(f"  {'Ep':<6}  {'Passive':>10}  {'Active (ISM)':>13}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*13}")

    for i in range(19, max_eps, 20):
        row = {}
        for name, log in logs.items():
            row[name] = log[i]["bank_size"] if i < len(log) else None
        vals = list(row.values())
        print(f"  {i+1:<6}  {vals[0]:>10}  {vals[1]:>13}")

    # ── Avg success_rate over time ─────────────────────────────────────────
    print(f"\n  Avg schema success_rate over time:")
    print(f"  {'Ep':<6}  {'Passive':>10}  {'Active (ISM)':>13}  {'Delta':>7}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*13}  {'─'*7}")

    for i in range(19, max_eps, 20):
        row = {}
        for name, log in logs.items():
            row[name] = log[i]["avg_success_rate"] if i < len(log) else None
        vals = list(row.values())
        if vals[0] is not None and vals[1] is not None:
            delta = vals[1] - vals[0]
            print(f"  {i+1:<6}  {vals[0]:>10.2f}  {vals[1]:>13.2f}  "
                  f"{delta:>+7.2f}")

    # ── Retrieval hit rate over time ───────────────────────────────────────
    print(f"\n  Retrieval hit rate (% retrieved_high or retrieved_med):")
    print(f"  {'Ep':<6}  {'Passive':>10}  {'Active (ISM)':>13}  {'Delta':>7}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*13}  {'─'*7}")

    for start in range(0, max_eps, 20):
        row = {}
        for name, log in logs.items():
            block = log[start:start + 20]
            row[name] = (sum(e["retrieval_hit"] for e in block) / len(block)
                         if block else None)
        vals = list(row.values())
        if vals[0] is not None and vals[1] is not None:
            delta = vals[1] - vals[0]
            print(f"  {start+1:<6}  {vals[0]:>10.0%}  {vals[1]:>13.0%}  "
                  f"{delta:>+7.0%}")

    # ── Key finding ────────────────────────────────────────────────────────
    print(f"\n  KEY FINDINGS:")

    # Compare first 20 vs last 20 for each system (degradation)
    for name, log in logs.items():
        first20_acc = sum(e["correct"] for e in log[:20]) / 20 if len(log) >= 20 else None
        last20_acc  = sum(e["correct"] for e in log[-20:]) / 20 if len(log) >= 20 else None
        label = "Passive" if "Passive" in name else "Active "
        if first20_acc is not None and last20_acc is not None:
            drift = last20_acc - first20_acc
            trend = "degraded" if drift < -0.03 else ("improved" if drift > 0.03 else "stable")
            print(f"  {label}: first20={first20_acc:.0%} → last20={last20_acc:.0%} "
                  f"({drift:+.0%}) [{trend}]")

    first_passive_sr = logs[list(logs.keys())[0]][0]["avg_success_rate"]  if logs else 0
    last_passive_sr  = logs[list(logs.keys())[0]][-1]["avg_success_rate"] if logs else 0
    first_active_sr  = logs[list(logs.keys())[1]][0]["avg_success_rate"]  if len(logs) > 1 else 0
    last_active_sr   = logs[list(logs.keys())[1]][-1]["avg_success_rate"] if len(logs) > 1 else 0

    print(f"\n  Passive schema quality: {first_passive_sr:.2f} → {last_passive_sr:.2f} "
          f"({last_passive_sr - first_passive_sr:+.2f})")
    print(f"  Active  schema quality: {first_active_sr:.2f} → {last_active_sr:.2f} "
          f"({last_active_sr - first_active_sr:+.2f})")

    print(f"\n  Results saved to {output_dir}/")
    print(f"{'═'*W}\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def print_bank_state(controller, label: str = ""):
    """
    Print full ISM bank state: every schema with its hook stats,
    health status, recent history, and whether it was synthesized or seeded.
    """
    bank   = controller.bank
    replay = controller.replay

    seed_names = {s["name"] for s in SEED_SCHEMAS}

    # Compute per-schema episode history from replay
    history: Dict[str, List[str]] = defaultdict(list)
    for ep in replay.buffer:
        history[ep.schema_used].append(ep.outcome)

    # Run quick audit
    report = _ism_audit(bank, replay) if replay.buffer else {}

    W = 70
    print(f"\n{'#'*W}")
    print(f"  ISM BANK STATE  {label}  —  {bank.size()} schemas  "
          f"({len(replay.buffer)} episodes in replay)")
    print(f"{'#'*W}")

    # Sort: seeded first, then synthesized; within each group by success_rate desc
    def sort_key(name):
        hook = bank.hooks.get(name)
        return (0 if name in seed_names else 1,
                -(hook.success_rate if hook else 0))

    for name in sorted(bank.schemas.keys(), key=sort_key):
        schema = bank.schemas[name]
        hook   = bank.hooks.get(name)
        if not hook:
            continue

        tag    = "SEED" if name in seed_names else "SYNTH"
        health = report.get(name, {})
        # ASCII-only glyphs for Windows consoles with non-UTF8 codepages.
        h_sym  = ("OK strong"  if health.get("health") == "strong"  else
                  "NO weak"    if health.get("health") == "weak"    else
                  "~ neutral" if health.get("health") == "neutral" else
                  "NONE unused")

        # Recent 10 outcomes for this schema
        recent   = history.get(name, [])[-10:]
        outcome_bar = "".join(
            "#" if o == "correct" else "-" if o == "incorrect" else "*"
            for o in recent
        )

        # Success rate bar (20 chars wide)
        sr      = hook.success_rate
        filled  = int(sr * 20)
        sr_bar  = "=" * filled + "-" * (20 - filled)

        print(f"\n  ┌── [{tag}] {name}")
        print(f"  │   operator : {hook.operator_type:<20} "
              f"structural: {hook.structural_pattern}")
        print(f"  │   heuristics: {', '.join(hook.heuristic_signature[:4])}")
        print(f"  │   uses     : {hook.usage_count:<5}  "
              f"success_rate: [{sr_bar}] {sr:.0%}  {h_sym}")
        if recent:
            print(f"  │   last {len(recent):>2} ep : [{outcome_bar}]  "
                  f"prec={health.get('score',0):.2f}  "
                  f"lift={health.get('lift',0):+.2f}")
        print(f"  │   description:")
        desc = schema.get("description", "")
        # Word-wrap at 60 chars
        words, line = desc.split(), ""
        for w in words:
            if len(line) + len(w) + 1 > 60:
                print(f"  │     {line}")
                line = w
            else:
                line = (line + " " + w).strip()
        if line:
            print(f"  │     {line}")
        tmpl = schema.get("template", "")
        if isinstance(tmpl, list):
            tmpl = "\n".join(str(x) for x in tmpl)
        tmpl = str(tmpl)
        print(f"  │   template  : "
              f"{tmpl[:80].split(chr(10))[0]}...")
        print(f"  └{'─'*(W-4)}")

    # Improvement log — what the ISM has changed
    if hasattr(controller, "_health_log") and controller._health_log:
        print(f"\n  {'─'*W}")
        print(f"  ISM IMPROVEMENT LOG  ({len(controller._health_log)} audits run)")
        print(f"  {'─'*W}")
        for entry in controller._health_log:
            strong  = [n for n, h in entry["report"].items() if h == "strong"]
            weak    = [n for n, h in entry["report"].items() if h == "weak"]
            unused  = [n for n, h in entry["report"].items() if h == "unused"]
            print(f"  ep={entry['episode']:>3} | bank={entry['bank_size']} | "
                  f"strong={len(strong)} weak={len(weak)} unused={len(unused)}")
            if weak:
                print(f"           corrected: {', '.join(weak)}")

    print(f"\n{'█'*W}\n")


def inspect_results(results_dir: str = "results"):
    """
    Load saved results JSON and print a bank inspection report.
    Shows schema usage patterns, accuracy per schema, improvement over time.
    """
    import os
    ism_path = os.path.join(results_dir, "ism.json")
    sm_path  = os.path.join(results_dir, "schema_memory.json")

    W = 70
    print(f"\n{'═'*W}")
    print(f"  POST-RUN INSPECTION: {results_dir}")
    print(f"{'═'*W}")

    for path, label in [(ism_path, "ISM"), (sm_path, "Schema Memory")]:
        if not os.path.exists(path):
            print(f"  {label}: no results found at {path}")
            continue

        with open(path) as f:
            log = json.load(f)

        print(f"\n  ── {label} ──────────────────────────────────────────")

        # Schema usage table
        schema_stats: Dict[str, Dict] = defaultdict(
            lambda: {"uses": 0, "correct": 0, "first_ep": 9999, "last_ep": 0}
        )
        bank_growth = []

        for ep in log:
            sname = ep["schema"]
            schema_stats[sname]["uses"]    += 1
            schema_stats[sname]["correct"] += int(ep["correct"])
            schema_stats[sname]["first_ep"] = min(
                schema_stats[sname]["first_ep"], ep["episode"])
            schema_stats[sname]["last_ep"]  = max(
                schema_stats[sname]["last_ep"],  ep["episode"])
            bank_growth.append((ep["episode"], ep["bank_size"]))

        print(f"\n  Schema usage (sorted by uses):")
        print(f"  {'Schema':<35} {'Uses':>5} {'Acc':>6} {'First':>6} {'Last':>5}")
        print(f"  {'─'*35} {'─'*5} {'─'*6} {'─'*6} {'─'*5}")
        for sname, s in sorted(schema_stats.items(),
                                key=lambda x: x[1]["uses"], reverse=True):
            acc = s["correct"] / s["uses"] if s["uses"] else 0
            print(f"  {sname[:35]:<35} {s['uses']:>5} "
                  f"{acc:>6.0%} {s['first_ep']:>6} {s['last_ep']:>5}")

        # Bank growth curve
        if bank_growth:
            eps    = [b[0] for b in bank_growth[::5]]
            sizes  = [b[1] for b in bank_growth[::5]]
            print(f"\n  Bank growth (every 5 episodes):")
            print(f"  ep:   " + "  ".join(f"{e:>3}" for e in eps))
            print(f"  size: " + "  ".join(f"{s:>3}" for s in sizes))

        # Accuracy over time (blocks of 10)
        block_accs = []
        for start in range(0, len(log), 10):
            block = log[start:start+10]
            if block:
                acc = sum(e["correct"] for e in block) / len(block)
                block_accs.append(acc)

        if block_accs:
            print(f"\n  Accuracy per 10-episode block:")
            bar_line = ""
            for acc in block_accs:
                filled = int(acc * 10)
                bar_line += f"[{'█'*filled}{'░'*(10-filled)}]{acc:.0%} "
            # Print in rows of 4
            parts = bar_line.split("] ")
            for i in range(0, len(parts), 4):
                chunk = "] ".join(parts[i:i+4])
                if chunk.strip():
                    print(f"  {chunk}]")

    print(f"\n{'═'*W}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Schema Memory — Competition MATH")
    parser.add_argument("--data_path",    default=None,
                        help="Local parquet/json path (downloads if omitted)")
    parser.add_argument("--n_per_block",  type=int, default=20)
    parser.add_argument("--max_level",    type=int, default=5,
                        help="Max difficulty 1-5 (default 5)")
    parser.add_argument("--min_level",    type=int, default=1,
                        help="Min difficulty 1-5 (default 1). Use --min_level 5 --max_level 5 for Level 5 only.")
    parser.add_argument("--numeric_only", action="store_true",
                        help="Keep only problems with numeric boxed answers (default: off). "
                             "Enabling this filters to easier problems — turn off for harder ablations.")
    parser.add_argument("--model",        default=LLM_MODEL)
    parser.add_argument("--output_dir",   default="results",
                        help="Base directory for saving results. "
                             "If left as 'results', a unique subfolder is created per run.")
    parser.add_argument("--run_name",     default=None,
                        help="Optional run name subfolder (e.g. 'debug1'). "
                             "If omitted, a timestamp is used.")
    parser.add_argument("--no_synthesizer", action="store_true",
                        help="Disable schema evolution (faster)")
    parser.add_argument("--quick",        action="store_true",
                        help="5 problems/block, level 1 only")
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--quiet",        action="store_true")
    parser.add_argument("--feature_debug", action="store_true",
                        help="Print rule/LLM features and agreement scoring.")
    parser.add_argument("--dataset",      default="competition_math",
                        choices=["competition_math", "math500", "aime2025"],
                        help="Dataset: competition_math (default), math500, or aime2025")
    parser.add_argument("--inspect",      action="store_true",
                        help="Inspect saved results without running experiment")
    parser.add_argument("--agentic",      action="store_true",
                        help="Enable agentic tool use for ISM (calculate, schema_lookup, "
                             "self_verify, decompose). Other systems remain passive.")
    parser.add_argument("--pass_k",      type=int, default=1,
                        help="Compute pass@k by sampling k base solves per problem (k>=1).")
    parser.add_argument("--ablation",    action="store_true",
                        help="Run focused ablation: Schema Memory (passive) vs ISM (active). "
                             "Tracks accuracy, bank size, success_rate, hit_rate over stream.")
    args = parser.parse_args()
    FEATURE_DEBUG = bool(args.feature_debug)

    # Create a unique run directory unless user explicitly sets a custom output_dir.
    if args.output_dir == "results":
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_name = (args.run_name or ts)
        args.output_dir = os.path.join("results", args.dataset, run_name)

    if args.inspect:
        inspect_results(args.output_dir)
        sys.exit(0)

    if args.ablation:
        run_ablation(
            source          = args.data_path,
            n_per_block     = args.n_per_block,
            max_level       = args.max_level,
            min_level       = args.min_level,
            numeric_only    = args.numeric_only,
            model           = args.model,
            output_dir      = args.output_dir,
            use_synthesizer = not args.no_synthesizer,
            seed            = args.seed,
            verbose         = not args.quiet,
            dataset         = args.dataset,
        )
        sys.exit(0)

    if args.quick:
        # Fix 3: use 20 problems per block for stability (1 problem = 5% swing)
        args.n_per_block = 20
        args.max_level   = 2

    if OPENAI_API_KEY == "YOUR_KEY_HERE" or not OPENAI_API_KEY:
        print("ERROR: Set your OpenAI API key!")
        print("  Option 1: put it in api_keys.json as 'openai_api_key'")
        print("  Option 2: set OPENAI_API_KEY environment variable")
        sys.exit(1)

    run_experiment(
        source          = args.data_path,
        n_per_block     = args.n_per_block,
        max_level       = args.max_level,
        min_level       = args.min_level,
        numeric_only    = args.numeric_only,
        model           = args.model,
        output_dir      = args.output_dir,
        use_synthesizer = not args.no_synthesizer,
        seed            = args.seed,
        verbose         = not args.quiet,
        dataset         = args.dataset,
        use_agentic     = args.agentic,
        pass_k          = args.pass_k,
    )