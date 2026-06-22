"""
baseline_reflexion.py — Reflexion Baseline

Based on: Shinn et al. "Reflexion: Language Agents with Verbal Reinforcement Learning"
NeurIPS 2023. https://arxiv.org/abs/2303.11366

After each episode, the LLM generates a verbal reflection on what went
wrong / right. Reflections are stored and retrieved for similar future
problems. No structured schema — just free-text verbal memory.

Key difference from ISM:
  - Reflexion stores REFLECTIONS (what went wrong in language)
  - ISM stores STRATEGIES (how to approach a problem class)
  - Reflexion is episode-level memory; ISM is schema-level memory
  - Reflexion memory never self-improves; ISM bank self-audits/merges/prunes
"""

import re
import json
import numpy as np
from typing import Optional
from openai import OpenAI

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


# ── Prompts ───────────────────────────────────────────────────────────────────

REFLECT_PROMPT = """You just attempted a math problem and got it {outcome}.

Problem:
{problem}

Your response:
{response}

Gold answer: {gold}

Write a SHORT verbal reflection (2-4 sentences) that:
1. Identifies the key mistake (if wrong) or key insight (if correct).
2. States what approach to use next time for this type of problem.
3. Notes any common traps to avoid.

Reflection:"""

REFLEXION_SYSTEM_PROMPT = """You are a precise mathematical problem solver.
You have ac,  read them carefully
to avoid repeating mistakes.

Output EXACTLY:

Steps to follow:
1. [plan incorporating lessons from reflections]

Solution:
- Step-by-step with units.

Sanity check:
- Brief justification.

Final Answer: <single value>"""

REFLECTION_TOP_K  = 3
REFLECTION_MIN_SIM = 0.50
MAX_REFLECTIONS    = 300   # cap to control memory


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-9 else 0.0


class ReflexionBaseline:
    """
    Verbal reinforcement learning baseline.
    Generates a reflection after each episode and retrieves relevant
    reflections before solving new problems.
    """

    name = "Reflexion"

    def __init__(self, encoder, verbose: bool = False):
        self.encoder   = encoder
        self.verbose   = verbose
        self._ep       = 0
        self.replay    = ReplayBuffer()
        self.extractor = HybridFeatureExtractor(llm_client=None)
        self.bank      = _ReflexionBank()
        self._client   = OpenAI(api_key=OPENAI_API_KEY)
        self._reflections: list[dict] = []   # verbal memory store
        self._current_emb = None

    # ── Reflection generation ─────────────────────────────────────────────────

    def _generate_reflection(self, problem: str, response: str,
                              gold: str, correct: bool) -> str:
        """Ask LLM to reflect on this episode."""
        try:
            prompt = REFLECT_PROMPT.format(
                outcome="CORRECTLY" if correct else "INCORRECTLY",
                problem=problem[:600],
                response=response[:600] if response else "(no response)",
                gold=gold,
            )
            resp = self._client.chat.completions.create(
                **_llm_kwargs(LLM_MODEL, max_tokens=200),
                messages=[
                    {"role": "system",
                     "content": "You are a math tutor reflecting on a student's solution."},
                    {"role": "user", "content": prompt},
                ],
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            return f"(reflection failed: {e})"

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def _retrieve_reflections(self, query_emb: np.ndarray) -> list[dict]:
        if not self._reflections:
            return []
        scored = []
        for ref in self._reflections:
            sim = cosine_sim(query_emb, ref["embedding"])
            if sim >= REFLECTION_MIN_SIM:
                scored.append((sim, ref))
        scored.sort(key=lambda x: -x[0])
        return [r for _, r in scored[:REFLECTION_TOP_K]]

    def _build_prompt(self, problem_text: str,
                      reflections: list[dict]) -> str:
        if not reflections:
            return problem_text

        parts = ["=== Lessons from Similar Past Problems ===\n"]
        for i, ref in enumerate(reflections, 1):
            status = "✓ correct" if ref["correct"] else "✗ mistake"
            parts.append(
                f"--- Reflection {i} [{ref.get('domain', 'Math')}] ({status}) ---\n"
                f"{ref['text']}\n"
            )
        parts.append("\n=== Problem to Solve ===\n")
        parts.append(problem_text)
        return "\n".join(parts)

    # ── Interface ─────────────────────────────────────────────────────────────

    def get_schema_for_problem(self, problem_text: str):
        emb      = self.encoder.encode(problem_text)
        features = self.extractor.extract(problem_text, emb)
        self._current_emb = emb

        reflections = self._retrieve_reflections(emb)
        self._current_reflections = reflections

        if self.verbose and reflections:
            print(f"  [Reflexion] Retrieved {len(reflections)} reflections")

        # Embed reflections in schema template for the solver
        reflexion_schema = {
            "name":          "Reflexion",
            "description":   f"Retrieved {len(reflections)} verbal reflections.",
            "template":      self._build_prompt(problem_text, reflections),
            "heuristics":    [],
            "_reflexion":    True,
            "_reflections":  reflections,
        }
        return reflexion_schema, features, float(len(reflections)) / REFLECTION_TOP_K, {}, (
            "retrieved_high" if reflections else "generic"
        )

    def solve(self, problem_text: str, schema: dict) -> str:
        """Solve with retrieved reflections as context."""
        augmented = self._build_prompt(
            problem_text, schema.get("_reflections", [])
        )
        resp = self._client.chat.completions.create(
            **_llm_kwargs(LLM_MODEL, max_tokens=700),
            messages=[
                {"role": "system", "content": REFLEXION_SYSTEM_PROMPT},
                {"role": "user",   "content": augmented},
            ],
        )
        return resp.choices[0].message.content or ""

    def after_episode(self, problem_text, features, schema_used,
                      ret_status, outcome, task_id,
                      tool_trace=None, response_text=None):
        correct = outcome == "correct"
        gold    = ""  # gold not passed here — reflection uses response only

        # Generate verbal reflection
        reflection_text = self._generate_reflection(
            problem=problem_text,
            response=response_text or "",
            gold=gold,
            correct=correct,
        )

        if self.verbose:
            print(f"  [Reflexion] {'✓' if correct else '✗'} "
                  f"Reflection: {reflection_text[:100]}...")

        # Store reflection with embedding of the problem
        self._reflections.append({
            "text":      reflection_text,
            "embedding": self._current_emb if self._current_emb is not None
                         else self.encoder.encode(problem_text),
            "correct":   correct,
            "domain":    task_id,
            "episode":   self._ep,
        })
        self.bank._count = len(self._reflections)

        # Cap size
        if len(self._reflections) > MAX_REFLECTIONS:
            self._reflections.pop(0)

        ep = Episode(
            episode_id=self._ep,
            problem_text=problem_text,
            features=features,
            schema_used="Reflexion",
            outcome=outcome,
            retrieval_score=0.0,
            task_id=task_id,
            log_llm_used=False,
        )
        self.replay.add(ep)
        self._ep += 1


class _ReflexionBank:
    def __init__(self):
        self.schemas = {}
        self.hooks   = {}
        self._count  = 0
    def size(self):
        return self._count


if __name__ == "__main__":
    print("ReflexionBaseline — import into run_comparison.py to use.")
