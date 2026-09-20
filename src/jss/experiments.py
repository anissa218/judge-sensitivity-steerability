"""Experiment matrix: the set of judge-call conditions for a dataset.

Each ``Condition`` fully specifies how to render a sample and how to score it
(which definition GT to use). The runner iterates conditions x seeds x
samples x judges.

Experiments
-----------
sens_context  (sensitivity to novel context; five_category w/ context only)
    context_none  (base) | context_correct | context_shuffled
steer_safety  (steerability via safety-policy edits)
    def_<name> for STEERABILITY_DEFINITIONS[family] (+ def_none for five_category)
steer_classification (same definitions, but reframed as A/B classification)
    cls_<name>
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import definitions as D


@dataclass
class Condition:
    experiment: str            # sens_context | steer_safety | steer_classification
    condition: str             # unique human-readable label within the dataset
    task: str                  # "safety" | "classification"
    definition: str            # definitions name used for gold + template
    with_context: bool = False
    context_mode: str = "none" # none | correct | shuffled


# Which classification definitions each family exposes (templates exist for these).
_CLS_DEFS = {
    "five_category": ["base", "strict", "swapped"],
    "sorry_bench": ["explicit", "definition_a", "definition_b"],
    "sports": [],  # no classification framing for the sports toy task
    "custom": ["base", "strict", "swapped"],
}

# Families whose policy is described over (category, severity) and which support
# the context experiment and the no-policy ablation.
_CATEGORY_FAMILIES = ("five_category", "custom")


def base_condition_for(experiment: str, family: str) -> str:
    """The reference condition each experiment's flip-rate is measured against."""
    if experiment == "sens_context":
        return "context_none"
    if experiment == "steer_safety":
        return f"def_{D.BASE_DEFINITION[family]}"
    if experiment == "steer_classification":
        defs = _CLS_DEFS[family]
        if not defs:
            raise ValueError(f"Family {family!r} has no classification definitions.")
        return f"cls_{defs[0]}"
    raise ValueError(f"Unknown experiment {experiment!r}")


def build_conditions(
    eval_df: pd.DataFrame,
    family: str,
    mode: str,
    experiments: list[str],
) -> list[Condition]:
    """Construct the condition list for the requested experiment groups.

    ``experiments`` may contain the umbrella names "sensitivity" /
    "steerability" or the fine-grained experiment names.
    """
    want = set(experiments)
    sens = {"sensitivity", "sens_context"} & want
    steer = {"steerability", "steer_safety", "steer_classification"} & want

    conditions: list[Condition] = []
    base_def = D.BASE_DEFINITION[family]

    # --- sensitivity: context (five_category + actual context only) ----------
    has_context = (eval_df["context"].fillna("").str.len() > 0).any()
    if (("sensitivity" in want) or ("sens_context" in want)) and family in _CATEGORY_FAMILIES and has_context:
        conditions.append(Condition("sens_context", "context_none", "safety", base_def, with_context=False, context_mode="none"))
        conditions.append(Condition("sens_context", "context_correct", "safety", base_def, with_context=True, context_mode="correct"))
        conditions.append(Condition("sens_context", "context_shuffled", "safety", base_def, with_context=True, context_mode="shuffled"))

    # --- steerability: safety-policy edits -----------------------------------
    if ("steerability" in want) or ("steer_safety" in want):
        defs = list(D.STEERABILITY_DEFINITIONS[family])
        if family in _CATEGORY_FAMILIES and "none" not in defs:
            defs = ["none"] + defs  # policy-ablation lives alongside the variants
        for name in defs:
            conditions.append(Condition("steer_safety", f"def_{name}", "safety", name))

    # --- steerability: classification reframe --------------------------------
    if (("steerability" in want) or ("steer_classification" in want)) and _CLS_DEFS[family]:
        for name in _CLS_DEFS[family]:
            conditions.append(Condition("steer_classification", f"cls_{name}", "classification", name))

    return conditions


def context_values(eval_df: pd.DataFrame, cond: Condition, seed: int) -> list[str]:
    """Per-sample context strings for a condition (handles global shuffling)."""
    ctx = eval_df["context"].fillna("").astype(str).tolist()
    if cond.context_mode == "correct":
        return ctx
    if cond.context_mode == "shuffled":
        rng = np.random.default_rng(seed)
        return list(rng.permutation(np.array(ctx, dtype=object)))
    return [""] * len(eval_df)  # "none"
