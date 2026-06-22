"""
smoke_test.py — Offline smoke test for ISM components.

Exercises the pure-Python building blocks of `schema_memory_agentic.py`
without making any OpenAI API calls. Use this to confirm that imports
resolve, dataclasses construct cleanly, the rule-based feature extractor
runs, the schema bank retrieves correctly, the replay buffer samples,
and the answer evaluator parses gold/predicted forms as expected.

Run:
    python smoke_test.py

Exit code 0 indicates all checks pass. Non-zero indicates a regression.
"""

import sys
import traceback

import numpy as np

FAILURES: list = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def section(title: str) -> None:
    print(f"\n-- {title} {'-' * max(0, 60 - len(title))}")


def main() -> int:
    # ── 1. Imports resolve ────────────────────────────────────────────
    section("Imports")
    try:
        from schema_memory_agentic import (
            FeatureExtractor,
            FeatureHook,
            ProblemFeatures,
            QuantitySignature,
            SchemaBank,
            ReplayBuffer,
            Episode,
            CLTracker,
            MathProblem,
            cosine_sim,
            extract_boxed,
            extract_final_answer,
            evaluate_answer,
            build_warm_hook,
            retrieve_schema,
            SEED_SCHEMAS,
        )
        check("import schema_memory_agentic symbols", True)
    except Exception as exc:
        check("import schema_memory_agentic symbols", False, str(exc))
        traceback.print_exc()
        return 1

    try:
        from feature_extractor_llm import safe_extract_features  # noqa: F401
        check("import feature_extractor_llm.safe_extract_features", True)
    except Exception as exc:
        check("import feature_extractor_llm.safe_extract_features", False, str(exc))

    # ── 2. Rule-based feature extractor ───────────────────────────────
    section("Rule-based feature extraction")
    extractor = FeatureExtractor()
    fake_emb = np.ones(8, dtype=np.float32) / np.sqrt(8)

    feat_alg = extractor.extract("Solve for x: 2x + 3 = 11", fake_emb)
    check("algebraic operator detected",
          feat_alg.operator_type == "algebraic",
          f"got {feat_alg.operator_type!r}")

    feat_nt = extractor.extract("Find the GCD of 12 and 18.", fake_emb)
    check("number_theory operator detected",
          feat_nt.operator_type == "number_theory",
          f"got {feat_nt.operator_type!r}")

    feat_geo = extractor.extract(
        "A circle has radius 5. Find its area.", fake_emb,
    )
    check("geometric operator detected",
          feat_geo.operator_type == "geometric",
          f"got {feat_geo.operator_type!r}")

    check("heuristics list non-empty",
          len(feat_alg.heuristics) > 0,
          f"got {feat_alg.heuristics!r}")

    # ── 3. Cosine similarity ──────────────────────────────────────────
    section("Cosine similarity")
    v = np.array([1.0, 0.0], dtype=np.float32)
    w = np.array([0.0, 1.0], dtype=np.float32)
    check("cosine_sim orthogonal == 0", abs(cosine_sim(v, w)) < 1e-6)
    check("cosine_sim identical == 1", abs(cosine_sim(v, v) - 1.0) < 1e-6)

    # ── 4. Schema bank insertion + retrieval ──────────────────────────
    section("Schema bank")
    bank = SchemaBank()

    class StubEncoder:
        def encode(self, text, normalize_embeddings=True):
            return fake_emb

    encoder = StubEncoder()
    for schema in SEED_SCHEMAS[:3]:
        emb = encoder.encode(schema["description"])
        # Reuse the rule extractor for a deterministic offline run.
        feat = extractor.extract(schema["description"], emb)
        hook = build_warm_hook(schema, feat, encoder)
        bank.add(schema, hook)
    check("bank size == 3 after inserts", bank.size() == 3)

    # Retrieve using a feature object roughly matching the first seed.
    probe_text = SEED_SCHEMAS[0]["description"]
    probe_feat = extractor.extract(probe_text, encoder.encode(probe_text))
    hook, score, breakdown, status = retrieve_schema(
        probe_feat, list(bank.hooks.values())
    )
    check("retrieve_schema returns a hook", hook is not None,
          f"status={status} score={score:.3f}")
    check("retrieval status is valid",
          status in {"retrieved_high", "retrieved_med",
                     "near_miss", "generic"},
          f"got {status!r}")

    # ── 5. Replay buffer ──────────────────────────────────────────────
    section("Replay buffer")
    replay = ReplayBuffer(max_size=5)
    for i in range(7):
        replay.add(Episode(
            episode_id=i,
            problem_text=f"problem {i}",
            features=probe_feat,
            schema_used="Algebra",
            outcome="correct" if i % 2 == 0 else "incorrect",
            retrieval_score=0.5,
            task_id="t",
        ))
    check("replay capped at max_size",
          len(replay.buffer) == 5,
          f"got {len(replay.buffer)}")
    check("replay sample returns at most n",
          len(replay.sample(n=3)) <= 3)
    check("recent_accuracy in [0, 1]",
          0.0 <= replay.recent_accuracy() <= 1.0)

    # ── 6. Answer extraction + evaluation ─────────────────────────────
    section("Answer evaluator")
    sample_response = (
        "Steps to follow:\n1. Set up.\n2. Solve.\n\n"
        "Solution:\n- 2x = 8, so x = 4.\n\n"
        "Final Answer: 4\n"
    )
    check("extract_final_answer numeric",
          extract_final_answer(sample_response) == "4")

    check("extract_boxed simple", extract_boxed(r"x = \boxed{42}") == "42")
    check("extract_boxed nested",
          extract_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}")

    check("evaluate_answer numeric match",
          evaluate_answer(sample_response, "4", 4.0) is True)
    check("evaluate_answer numeric mismatch",
          evaluate_answer(sample_response, "5", 5.0) is False)
    check("evaluate_answer empty response",
          evaluate_answer("", "4", 4.0) is False)

    # ── 7. CL metrics tracker ─────────────────────────────────────────
    section("CL metrics")
    tracker = CLTracker()
    for _ in range(5):
        tracker.record("task_a", True)
    for _ in range(3):
        tracker.record("task_b", False)
    metrics = tracker.compute()
    check("CLTracker computes average_accuracy",
          "average_accuracy" in metrics)
    check("CLTracker accuracy in [0, 1]",
          0.0 <= metrics["average_accuracy"] <= 1.0)

    # ── 8. Dataclass round-trip ───────────────────────────────────────
    section("Dataclasses")
    qs = QuantitySignature(n_known=2, has_rate=True, unit_type="time")
    check("QuantitySignature similarity self == 1.0",
          abs(qs.similarity(qs) - 1.0) < 1e-6)

    prob = MathProblem(
        problem_id=0, text="x+1=2", level=1, type="Algebra",
        solution=r"x = \boxed{1}", boxed_answer="1", numeric_answer=1.0,
    )
    check("MathProblem.__post_init__ sets task_id",
          prob.task_id == "Algebra")

    # ── Summary ───────────────────────────────────────────────────────
    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {', '.join(FAILURES)}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
