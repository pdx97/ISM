"""
baseline_rag_examples.py — RAG-over-Examples Baseline

Retrieves the most similar past solved problems (with their solutions) and
injects them as few-shot examples in the prompt. No schema, no strategy —
just "here are similar problems that were solved before, now solve this one."

This is the critical baseline: it answers the reviewer question
"why not just use RAG?" If ISM > RAG, strategies beat examples.
If ISM ≈ RAG, schemas add nothing over raw examples.

Memory structure:
  EpisodeStore: list of {problem, solution, correct, domain, embedding}
  Retrieval: cosine similarity on embeddings → top-k
  Injection: few-shot examples prepended to the problem
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
        extract_final_answer,
    )
except ImportError as e:
    raise ImportError(f"Could not import from schema_memory_agentic.py: {e}")


RAG_SYSTEM_PROMPT = """You are a precise mathematical problem solver.
You will be given similar solved problems as examples to guide your approach.
Study the examples, then solve the new problem using the same style.

Output EXACTLY:

Steps to follow:
1. [plan based on examples]

Solution:
- Step-by-step with units.

Sanity check:
- Brief justification.

Final Answer: <single value>"""

RAG_TOP_K      = 3     # number of examples to retrieve
RAG_MIN_SIM    = 0.50  # minimum similarity to include an example
RAG_MAX_STORE  = 500   # cap the example store size to control memory


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-9 else 0.0


class RAGExamplesBaseline:
    """
    Retrieval-Augmented Generation over past solved examples.
    Stores (problem, solution, outcome, embedding) tuples.
    Retrieves top-k by embedding similarity and injects as few-shot context.
    """

    name = "RAG-over-Examples"

    def __init__(self, encoder, verbose: bool = False):
        self.encoder   = encoder
        self.verbose   = verbose
        self._ep       = 0
        self.replay    = ReplayBuffer()
        self.extractor = HybridFeatureExtractor(llm_client=None)
        self.bank      = _RAGBank()        # tracks example count
        self._client   = OpenAI(api_key=OPENAI_API_KEY)
        self._store: list[dict] = []       # the example memory

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def _retrieve_examples(self, query_emb: np.ndarray) -> list[dict]:
        if not self._store:
            return []
        scored = []
        for ex in self._store:
            sim = cosine_sim(query_emb, ex["embedding"])
            if sim >= RAG_MIN_SIM:
                scored.append((sim, ex))
        scored.sort(key=lambda x: -x[0])
        return [ex for _, ex in scored[:RAG_TOP_K]]

    def _build_prompt(self, problem_text: str, examples: list[dict]) -> str:
        if not examples:
            return problem_text

        parts = ["=== Similar Solved Problems (use these as reference) ===\n"]
        for i, ex in enumerate(examples, 1):
            status = "✓ Correct" if ex["correct"] else "✗ Incorrect (avoid this approach)"
            parts.append(
                f"--- Example {i} [{ex.get('domain', 'Math')}] {status} ---\n"
                f"Problem: {ex['problem'][:400]}\n"
                f"Solution approach: {ex['solution'][:600]}\n"
                f"Answer: {ex['answer']}\n"
            )
        parts.append("\n=== New Problem to Solve ===\n")
        parts.append(problem_text)
        return "\n".join(parts)

    # ── Interface ─────────────────────────────────────────────────────────────

    def get_schema_for_problem(self, problem_text: str):
        emb      = self.encoder.encode(problem_text)
        features = self.extractor.extract(problem_text, emb)

        examples = self._retrieve_examples(emb)
        # Store retrieved examples in features for use by solve()
        self._current_emb      = emb
        self._current_examples = examples

        if self.verbose and examples:
            print(f"  [RAG] Retrieved {len(examples)} examples "
                  f"(sim={cosine_sim(emb, examples[0]['embedding']):.3f})")

        # Return a pseudo-schema containing the few-shot prompt
        # The comparison runner calls hard_reset_call(problem, schema) —
        # we embed the RAG context in the schema template field
        rag_schema = {
            "name":        "RAG-over-Examples",
            "description": f"Retrieved {len(examples)} similar solved problems.",
            "template":    self._build_prompt(problem_text, examples),
            "heuristics":  [],
            "_rag_prompt": True,   # flag for custom solve path
            "_examples":   examples,
        }
        return rag_schema, features, float(len(examples)) / RAG_TOP_K, {}, (
            "retrieved_high" if examples else "generic"
        )

    def solve(self, problem_text: str, schema: dict) -> str:
        """
        Custom solve: inject retrieved examples into the prompt directly.
        This bypasses the schema template injection in hard_reset_call.
        """
        augmented_problem = self._build_prompt(
            problem_text, schema.get("_examples", [])
        )
        resp = self._client.chat.completions.create(
            **_llm_kwargs(LLM_MODEL, max_tokens=800),
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user",   "content": augmented_problem},
            ],
        )
        return resp.choices[0].message.content or ""

    def after_episode(self, problem_text, features, schema_used,
                      ret_status, outcome, task_id,
                      tool_trace=None, response_text=None):
        # Store this episode as a future example
        answer = ""
        if response_text:
            answer = extract_final_answer(response_text) or ""
            m = re.findall(r'\\boxed\{([^}]*)\}', response_text)
            if not answer and m:
                answer = m[-1]

        example = {
            "problem":   problem_text[:600],
            "solution":  response_text[:800] if response_text else "",
            "answer":    answer,
            "correct":   outcome == "correct",
            "domain":    task_id,
            "embedding": getattr(self, "_current_emb",
                                 self.encoder.encode(problem_text)),
        }
        self._store.append(example)
        self.bank._count = len(self._store)

        # Cap store size — remove oldest if over limit
        if len(self._store) > RAG_MAX_STORE:
            self._store.pop(0)

        ep = Episode(
            episode_id=self._ep,
            problem_text=problem_text,
            features=features,
            schema_used="RAG-over-Examples",
            outcome=outcome,
            retrieval_score=0.0,
            task_id=task_id,
            log_llm_used=False,
        )
        self.replay.add(ep)
        self._ep += 1


class _RAGBank:
    """Tracks example store size so comparison runner can log it."""
    def __init__(self):
        self.schemas = {}
        self.hooks   = {}
        self._count  = 0
    def size(self):
        return self._count


if __name__ == "__main__":
    print("RAGExamplesBaseline — import into run_comparison.py to use.")
