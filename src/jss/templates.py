"""Judge-prompt rendering.

Selects the appropriate template file for a given
(family, mode, task, definition, context) combination and fills the
``{conversation}`` / ``{generation}`` / ``{prompt_context}`` placeholders.

We fill ``{conversation}`` with the raw prompt text (no "User:" prefix).

``task`` is "safety" (YES-UNSAFE / NO-SAFE) or "classification" (A / B).
"""

from __future__ import annotations

from pathlib import Path

from . import policy as _policy

TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "templates"

# (family, mode, task, definition) -> filename. ``mode`` only matters for the
# five_category family; sorry_bench/sports are completion-only.
_FILES = {
    # ---- five_category, completion, safety -------------------------------
    ("five_category", "completion", "safety", "none"): "English_no_examples_no_context_no_definition.txt",
    ("five_category", "completion", "safety", "base"): "English_no_examples_no_context.txt",
    ("five_category", "completion", "safety", "base_context"): "English_no_examples_with_context.txt",
    ("five_category", "completion", "safety", "strict"): "English_no_examples_no_context_definition_b.txt",
    ("five_category", "completion", "safety", "swapped"): "English_no_examples_no_context_definition_a.txt",
    # ---- five_category, completion, classification -----------------------
    ("five_category", "completion", "classification", "base"): "English_no_examples_no_context_categorisation.txt",
    ("five_category", "completion", "classification", "strict"): "English_no_examples_no_context_definition_b_categorisation.txt",
    ("five_category", "completion", "classification", "swapped"): "English_no_examples_no_context_definition_a_categorisation.txt",
    # ---- five_category, prompt, safety -----------------------------------
    ("five_category", "prompt", "safety", "none"): "English_no_examples_no_context_no_definition_prompt_only.txt",
    ("five_category", "prompt", "safety", "base"): "English_prompt_only_no_context.txt",
    ("five_category", "prompt", "safety", "base_context"): "English_prompt_only_with_context.txt",
    ("five_category", "prompt", "safety", "strict"): "English_no_examples_no_context_definition_b_prompt_only.txt",
    ("five_category", "prompt", "safety", "swapped"): "English_no_examples_no_context_definition_a_prompt_only.txt",
    # ---- five_category, prompt, classification ---------------------------
    ("five_category", "prompt", "classification", "base"): "English_no_examples_no_context_categorisation_prompt_only.txt",
    ("five_category", "prompt", "classification", "strict"): "English_no_examples_no_context_definition_b_categorisation_prompt_only.txt",
    ("five_category", "prompt", "classification", "swapped"): "English_no_examples_no_context_definition_a_categorisation_prompt_only.txt",
    # ---- sorry_bench, completion, safety ---------------------------------
    ("sorry_bench", "completion", "safety", "compliance"): "English_sorry_bench_compliance.txt",
    ("sorry_bench", "completion", "safety", "no_definition"): "English_sorry_bench_no_definition.txt",
    ("sorry_bench", "completion", "safety", "explicit"): "English_sorry_bench_definition_explicit.txt",
    ("sorry_bench", "completion", "safety", "definition_a"): "English_sorry_bench_definition_a.txt",
    ("sorry_bench", "completion", "safety", "definition_b"): "English_sorry_bench_definition_b.txt",
    # ---- sorry_bench, completion, classification -------------------------
    ("sorry_bench", "completion", "classification", "explicit"): "English_sorry_bench_categorisation_definition_explicit.txt",
    ("sorry_bench", "completion", "classification", "definition_a"): "English_sorry_bench_categorisation_definition_a.txt",
    ("sorry_bench", "completion", "classification", "definition_b"): "English_sorry_bench_categorisation_definition_b.txt",
    # ---- sports, completion, safety --------------------------------------
    ("sports", "completion", "safety", "A"): "English_no_examples_no_context_ball_sports.txt",
    ("sports", "completion", "safety", "B"): "English_no_examples_no_context_ball_sports_a.txt",
    ("sports", "completion", "safety", "C"): "English_no_examples_no_context_ball_sports_b.txt",
    ("sports", "completion", "safety", "standard"): "English_no_examples_no_context.txt",
}

_cache: dict[str, str] = {}


def _load(filename: str) -> str:
    if filename not in _cache:
        path = TEMPLATES_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Template not found: {path}")
        _cache[filename] = path.read_text(encoding="utf-8")
    return _cache[filename]


def template_file(family: str, mode: str, task: str, definition: str, with_context: bool) -> str:
    """Resolve the template filename for the given combination."""
    key_def = definition
    if with_context and definition == "base":
        key_def = "base_context"
    key = (family, mode, task, key_def)
    if key not in _FILES:
        raise ValueError(f"No template for {key}. Known: see templates._FILES")
    return _FILES[key]


def render(
    *,
    family: str,
    mode: str,
    task: str,
    definition: str,
    prompt: str,
    completion: str = "",
    context: str = "",
    with_context: bool = False,
) -> str:
    """Render the full judge user-message for one sample."""
    if family == "custom":
        template = _policy.get_active_policy().generate(mode, task, definition, with_context)
    else:
        template = _load(template_file(family, mode, task, definition, with_context))

    return template.format(
        conversation=prompt,
        generation=completion,
        prompt_context=context or "",
    )
