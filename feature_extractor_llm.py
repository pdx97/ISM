"""
Feature extraction via LLM. Uses canonical taxonomy aligned with schema_memory_agentic.
"""
import json
import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Canonical taxonomy — must match OPERATOR_KEYWORDS / STRUCTURAL_KEYWORDS in schema_memory_agentic
# FEATURE_SCHEMA = {
#     "operator": [
#         "algebraic", "number_theory", "geometric",
#         "combinatoric", "probability", "calculus"
#     ],
#     "structure": [
#         "find_missing", "comparison", "two_agents_combined",
#         "part_whole", "before_after"
#     ],
#     "heuristics": [
#         "decompose", "work_backwards", "introduce_variable",
#         "find_pattern", "change_representation"
#     ],
# }
FEATURE_SCHEMA = {
    "operator": [
        "algebraic", "number_theory", "geometric",
        "combinatoric", "probability", "calculus"
    ],
    "structure": [
        "two_agents_combined", "part_whole", "before_after",
        "comparison", "optimization", "existence_count",
        "construction", "evaluate_expression", "find_missing"
    ],
    "heuristics": [
        "decompose", "work_backwards", "introduce_variable",
        "find_pattern", "change_representation",
        "examine_special_cases", "use_symmetry",
        "argue_by_contradiction", "apply_theorems", "visualize"
    ],
}
# Canonical operator labels -> seed schema names in schema_memory_agentic.py
OPERATOR_TO_SEED_SCHEMA = {
    "algebraic": "Algebra",
    "number_theory": "Number Theory",
    "geometric": "Geometry",
    "combinatoric": "Combinatorics",
    "probability": "Probability",
    "calculus": "Calculus and Analysis",
}


def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^\s*```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


def call_llm(prompt: str, model, model_name: Optional[str] = None) -> str:
    """
    model: OpenAI client compatible with client.chat.completions.create.
    model_name: explicit model (e.g. "gpt-4o-mini"). If None, infers from client or defaults.
    """
    if model is None:
        raise ValueError("LLM model/client is None")
    name = model_name or getattr(model, "_default_model", None) or getattr(model, "model", None)
    if not isinstance(name, str) or not name:
        name = "gpt-4o-mini"
    resp = model.chat.completions.create(
        model=name,
        temperature=0.0,
        max_completion_tokens=220,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content or ""


def extract_features_llm(problem_text: str, model, model_name: Optional[str] = None) -> dict:
    prompt = f"""
Extract structured features from the math problem.

Return ONLY valid JSON.

Problem:
{problem_text}

Schema:
- operator: one of {FEATURE_SCHEMA["operator"]}
- seed_schema: one of {list(OPERATOR_TO_SEED_SCHEMA.values())}
- structure_top2: list of 1-2 items from {FEATURE_SCHEMA["structure"]} (most likely first)
- heuristics: list of max 3 from {FEATURE_SCHEMA["heuristics"]}
- quantities:
    - has_rate (true/false)
    - has_time (true/false)
    - has_constraint (true/false)
    - num_values (integer)
    - n_unknown (integer)
- confidence: float between 0 and 1 (diagnostic only; caller may ignore)

Examples:
Example 1 input:
"If 2x + 3 = 11, find x."
Example 1 output:
{{
  "operator": "algebraic",
  "seed_schema": "Algebra",
  "structure_top2": ["find_missing"],
  "heuristics": ["introduce_variable", "decompose"],
  "quantities": {{
    "has_rate": false,
    "has_time": false,
    "has_constraint": false,
    "num_values": 2,
    "n_unknown": 1
  }},
  "confidence": 0.90
}}

Example 2 input:
"How many ways can 5 students be seated in 3 chairs?"
Example 2 output:
{{
  "operator": "combinatoric",
  "seed_schema": "Combinatorics",
  "structure_top2": ["comparison", "find_missing"],
  "heuristics": ["decompose", "find_pattern"],
  "quantities": {{
    "has_rate": false,
    "has_time": false,
    "has_constraint": true,
    "num_values": 2,
    "n_unknown": 1
  }},
  "confidence": 0.80
}}
"""

    response = call_llm(prompt, model, model_name)
    parsed = json.loads(_strip_code_fences(response))
    if not isinstance(parsed, dict):
        raise ValueError("LLM returned non-object JSON")
    return parsed


def validate_features(f: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(f, dict):
        raise ValueError("Features must be a dict")

    if f.get("operator") not in FEATURE_SCHEMA["operator"]:
        raise ValueError("Invalid operator")
    expected_seed = OPERATOR_TO_SEED_SCHEMA[f["operator"]]
    provided_seed = f.get("seed_schema")
    if provided_seed and str(provided_seed).strip() != expected_seed:
        raise ValueError("seed_schema inconsistent with operator")
    f["seed_schema"] = expected_seed

    # Accept either structure_top2 (preferred) or legacy single structure.
    raw_struct = f.get("structure_top2", f.get("structure", []))
    if isinstance(raw_struct, str):
        raw_struct = [raw_struct]
    if not isinstance(raw_struct, list):
        raw_struct = []
    struct = [s for s in raw_struct if s in FEATURE_SCHEMA["structure"]]
    if not struct:
        raise ValueError("Invalid structure")
    f["structure_top2"] = struct[:2]
    f["structure"] = f["structure_top2"][0]  # backward compatibility

    heur = f.get("heuristics", [])
    if not isinstance(heur, list):
        heur = []
    f["heuristics"] = [h for h in heur if h in FEATURE_SCHEMA["heuristics"]][:3]

    q = f.get("quantities", {})
    if not isinstance(q, dict):
        q = {}

    def _as_bool(x) -> bool:
        if isinstance(x, bool):
            return x
        if isinstance(x, (int, float)):
            return bool(x)
        if isinstance(x, str):
            return x.strip().lower() in ("true", "1", "yes", "y")
        return False

    def _as_int(x) -> int:
        if isinstance(x, bool):
            return int(x)
        if isinstance(x, int):
            return x
        if isinstance(x, float):
            return int(x)
        if isinstance(x, str):
            m = re.search(r"-?\d+", x)
            return int(m.group(0)) if m else 0
        return 0

    f["quantities"] = {
        "has_rate": _as_bool(q.get("has_rate", False)),
        "has_time": _as_bool(q.get("has_time", False)),
        "has_constraint": _as_bool(q.get("has_constraint", False)),
        "num_values": max(0, _as_int(q.get("num_values", 0))),
        "n_unknown": max(0, _as_int(q.get("n_unknown", 1))),
    }

    conf = f.get("confidence", 0.0)
    try:
        conf_f = float(conf)
    except Exception:
        conf_f = 0.0
    f["confidence"] = min(1.0, max(0.0, conf_f))

    return f


def safe_extract_features(
    problem_text: str,
    model,
    min_confidence: float = 0.6,
    model_name: Optional[str] = None,
    verbose: bool = False,
) -> Optional[Dict[str, Any]]:
    try:
        f = extract_features_llm(problem_text, model, model_name)
        f = validate_features(f)
        return f
    except Exception as e:
        logger.debug("safe_extract_features failed: %s", e, exc_info=True)
        if verbose:
            print(f"  [FeatureExtractor] LLM ERROR: {e}")
        return None
