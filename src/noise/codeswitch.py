import random
import re
from typing import Dict, List

EN_TO_FR: Dict[str, str] = {
    "the": "le",
    "a": "un",
    "an": "un",
    "and": "et",
    "or": "ou",
    "in": "dans",
    "on": "sur",
    "for": "pour",
    "with": "avec",
    "of": "de",
    "to": "à",
    "is": "est",
    "are": "sont",
    "was": "était",
    "were": "étaient",
    "where": "où",
    "when": "quand",
    "what": "quoi",
    "which": "quel",
    "year": "année",
    "city": "ville",
}

FR_TO_EN: Dict[str, str] = {
    "le": "the",
    "la": "the",
    "les": "the",
    "un": "a",
    "une": "a",
    "et": "and",
    "ou": "or",
    "dans": "in",
    "sur": "on",
    "pour": "for",
    "avec": "with",
    "de": "of",
    "à": "to",
    "est": "is",
    "sont": "are",
    "où": "where",
    "quand": "when",
    "quoi": "what",
    "quel": "which",
    "année": "year",
    "ville": "city",
    "projet": "project",
}

EN_INSERTIONS: List[str] = [
    "basically", "like", "so", "actually", "project", "team", "start", "city", "year"
]


def _tokenize_keep_punct(text: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)


def _is_word(tok: str) -> bool:
    return re.fullmatch(r"\w+", tok, re.UNICODE) is not None


def apply_codeswitch_noise(text: str, noise_p: float, language: str) -> str:
    """
    Code-switch noise: replace a small fraction of function words with an alternate language,
    or inject a small English token occasionally (for zh).

    - For 'en': swap some common words -> French
    - For 'fr': swap some common words -> English
    - For 'zh': inject occasional English tokens
    """
    if noise_p <= 0.0:
        return text

    rng = random.random

    if language == "zh":
        tokens = _tokenize_keep_punct(text)
        out: List[str] = []
        for tok in tokens:
            out.append(tok)
            if _is_word(tok) and rng() < (noise_p * 0.25):
                out.append(random.choice(EN_INSERTIONS))
        return _smart_join(out)

    tokens = _tokenize_keep_punct(text)
    out: List[str] = []

    if language == "en":
        mapping = EN_TO_FR
    elif language == "fr":
        mapping = FR_TO_EN
    else:
        return text

    for tok in tokens:
        if _is_word(tok) and rng() < noise_p:
            low = tok.lower()
            if low in mapping:
                repl = mapping[low]
                if tok[:1].isupper():
                    repl = repl[:1].upper() + repl[1:]
                out.append(repl)
                continue
        out.append(tok)

    return _smart_join(out)


def _smart_join(tokens: List[str]) -> str:
    """
    Join tokens with spaces, but avoid spaces before punctuation.
    """
    if not tokens:
        return ""

    out = tokens[0]
    for t in tokens[1:]:
        if re.fullmatch(r"[.,!?;:)\]\}]", t):
            out += t
        elif re.fullmatch(r"[(\[\{]", t):
            out += " " + t
        else:
            out += " " + t
    return out
