"""
baseline_vanilla.py — Vanilla LLM Baseline

No memory whatsoever. Every problem is solved fresh with just the system
prompt and no schema, no examples, no reflection. This is the lower bound —
shows what the base LLM can do without any memory mechanism.

Interface matches MemoryController so it plugs into run_comparison.py.
"""

import re
import numpy as np
from typing import Optional
from openai import OpenAI

# ---------------------------------------------------------------------------
# Import shared utilities from main module
# ---------------------------------------------------------------------------
try:
    from schema_memory_agentic import (
        OPENAI_API_KEY,
        LLM_MODEL,
        _llm_kwargs,
        GENERIC_SCHEMA,
        ProblemFeatures,
        Episode,
        ReplayBuffer,
        HybridFeatureExtractor,
    )
except ImportError as e:
    raise ImportError(f"Could not import from schema_memory_agentic.py: {e}")


VANILLA_SYSTEM_PROMPT = """You are a precise mathematical problem solver.
For every problem output EXACTLY:

Steps to follow:
1. [plan, 3-6 items]

Solution:
- Step-by-step with units.

Sanity check:
- Brief justification.

Final Answer: <single value>"""


class VanillaLLM:
    """
    No memory. Solves each problem independently with just the system prompt.
    bank attribute is a stub so the comparison runner can call bank.size().
    """

    name = "Vanilla LLM"

    def __init__(self, encoder, verbose: bool = False):
        self.encoder  = encoder
        self.verbose  = verbose
        self._ep      = 0
        self.replay   = ReplayBuffer()
        self.extractor = HybridFeatureExtractor(llm_client=None)
        self.bank     = _StubBank()
        self._client  = OpenAI(api_key=OPENAI_API_KEY)

    def get_schema_for_problem(self, problem_text: str):
        emb      = self.encoder.encode(problem_text)
        features = self.extractor.extract(problem_text, emb)
        # Always returns generic — no memory
        return GENERIC_SCHEMA, features, 0.0, {}, "generic"

    def solve(self, problem_text: str) -> str:
        """Direct LLM call with no schema injection."""
        resp = self._client.chat.completions.create(
            **_llm_kwargs(LLM_MODEL, max_tokens=600),
            messages=[
                {"role": "system", "content": VANILLA_SYSTEM_PROMPT},
                {"role": "user",   "content": problem_text},
            ],
        )
        return resp.choices[0].message.content or ""

    def after_episode(self, problem_text, features, schema_used,
                      ret_status, outcome, task_id,
                      tool_trace=None, response_text=None):
        ep = Episode(
            episode_id=self._ep,
            problem_text=problem_text,
            features=features,
            schema_used="Generic",
            outcome=outcome,
            retrieval_score=0.0,
            task_id=task_id,
            log_llm_used=False,
        )
        self.replay.add(ep)
        self._ep += 1


class _StubBank:
    """Minimal stub so comparison runner can call bank.size() and bank.hooks."""
    def __init__(self):
        self.schemas = {}
        self.hooks   = {}
    def size(self):
        return 0


if __name__ == "__main__":
    print("VanillaLLM baseline — import into run_comparison.py to use.")
