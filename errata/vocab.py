from __future__ import annotations

import json
import re

SYNONYMS_OVERRIDE_KEY = "synonyms_override"

IRREGULAR_PLURALS = {
    "person": ("people", "persons"),
    "mouse": ("mice",),
    "goose": ("geese",),
    "foot": ("feet",),
    "child": ("children",),
}


def label_variants(label: str) -> list[str]:
    return [label, label.replace("_", " ")]


def _suffix_pattern(word: str) -> str:
    lowered = word.lower()
    if lowered.endswith("y") and len(lowered) > 2 and lowered[-2] not in "aeiou":
        return re.escape(word[:-1]) + "(?:y|ies)"
    if lowered.endswith(("s", "x", "z", "ch", "sh")):
        return re.escape(word) + "(?:es)?"
    return re.escape(word) + "s?"


def label_in_text(label: str, text: str, synonyms: dict | None = None) -> bool:
    synonyms = synonyms or {}
    for variant in label_variants(label):
        words = [variant]
        words.extend(s for s in synonyms.get(variant.lower(), []) if s)
        alternates = []
        for word in words:
            alternates.append(_suffix_pattern(word))
            alternates.extend(
                _suffix_pattern(p) for p in IRREGULAR_PLURALS.get(word.lower(), ())
            )
        pattern = r"\b(?:" + "|".join(alternates) + r")(?:'s)?\b"
        if re.search(pattern, text, flags=re.IGNORECASE):
            return True
    return False


def effective_synonyms(config: dict, db) -> dict:
    """Baseline synonyms plus user additions (append, deduped).

    The DB override stores only additions; the packaged/config baseline is
    always preserved so built-in synonyms cannot be removed from the UI.
    """
    baseline = {
        k: list(v) if isinstance(v, list) else v
        for k, v in ((config.get("analysis", {}) or {}).get("synonyms", {}) or {}).items()
    }
    raw = db.kv_get(SYNONYMS_OVERRIDE_KEY)
    if raw is None:
        return baseline
    try:
        additions = json.loads(raw) or {}
    except json.JSONDecodeError:
        return baseline
    for label, values in additions.items():
        if not isinstance(values, list):
            continue
        existing = baseline.get(label)
        if not isinstance(existing, list):
            baseline[label] = list(values)
            continue
        seen = {str(v).strip().lower() for v in existing}
        for value in values:
            norm = str(value).strip().lower()
            if norm and norm not in seen:
                existing.append(value)
                seen.add(norm)
    return baseline
