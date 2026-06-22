"""
baseline_static_schema.py — Static Schema Baseline

One hand-crafted general math schema that never changes, never updates,
and is retrieved for every problem regardless of type.

Shows: does adaptive retrieval matter, or is a single good prompt enough?
If ISM ≈ Static, then the retrieval mechanism adds nothing.
If ISM > Static, the schema bank earns its complexity.
"""

import numpy as np
from typing import Optional
from openai import OpenAI

try:
    from schema_memory_agentic import (
        OPENAI_API_KEY,
        LLM_MODEL,
        _llm_kwargs,
        ProblemFeatures,
        Episode,
        ReplayBuffer,
        HybridFeatureExtractor,
        hard_reset_call,
    )
except ImportError as e:
    raise ImportError(f"Could not import from schema_memory_agentic.py: {e}")


# One carefully written general-purpose math schema.
# This is the best single schema a human expert would write — the ceiling
# for a static non-adaptive approach.
STATIC_SCHEMA = {
    "name": "General Mathematical Problem Solver",
    "description": (
        "A universal schema for competition mathematics. Covers algebra, "
        "geometry, number theory, combinatorics, and probability problems "
        "at difficulty levels 4-5."
    ),
    "template": (
        "1. READ: Identify all given quantities, constraints, and what is asked.\n"
        "2. CLASSIFY: Determine the mathematical domain (algebra / geometry / "
        "number theory / combinatorics / probability).\n"
        "3. PLAN: Choose a strategy (equation setup / coordinate geometry / "
        "modular arithmetic / counting principle / Bayes).\n"
        "4. EXECUTE: Carry out the solution step by step, tracking units.\n"
        "5. VERIFY: Substitute answer back or check boundary cases.\n"
        "6. SIMPLIFY: Express answer in simplest exact form."
    ),
    "heuristics": [
        "introduce_variable",
        "draw_diagram",
        "check_small_cases",
        "use_symmetry",
        "work_backwards",
        "decompose_into_sub_problems",
        "verify_by_substitution",
    ],
    "operator_type": "general",
}


class StaticSchemaBaseline:
    """
    Always returns the same fixed schema. No retrieval, no update.
    """

    name = "Static Schema"

    def __init__(self, encoder, verbose: bool = False):
        self.encoder   = encoder
        self.verbose   = verbose
        self._ep       = 0
        self.replay    = ReplayBuffer()
        self.extractor = HybridFeatureExtractor(llm_client=None)
        self.bank      = _StaticBank()

    def get_schema_for_problem(self, problem_text: str):
        emb      = self.encoder.encode(problem_text)
        features = self.extractor.extract(problem_text, emb)
        # Always the same schema — no retrieval logic
        return STATIC_SCHEMA, features, 1.0, {"static": 1.0}, "retrieved_high"

    def after_episode(self, problem_text, features, schema_used,
                      ret_status, outcome, task_id,
                      tool_trace=None, response_text=None):
        # No update — schema is static
        ep = Episode(
            episode_id=self._ep,
            problem_text=problem_text,
            features=features,
            schema_used="General Mathematical Problem Solver",
            outcome=outcome,
            retrieval_score=1.0,
            task_id=task_id,
            log_llm_used=False,
        )
        self.replay.add(ep)
        self._ep += 1


class _StaticBank:
    """Single-schema bank stub."""
    def __init__(self):
        self.schemas = {"General Mathematical Problem Solver": STATIC_SCHEMA}
        self.hooks   = {}
    def size(self):
        return 1


if __name__ == "__main__":
    print("StaticSchemaBaseline — import into run_comparison.py to use.")
