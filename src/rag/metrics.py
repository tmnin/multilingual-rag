import re


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\W+", " ", text)
    return text.strip()


def is_correct(pred: str, gold: str) -> bool:
    """
    Very simple exact/substring match.
    This is intentional for small controlled QA.
    """
    pred_n = normalize(pred)
    gold_n = normalize(gold)

    return gold_n in pred_n or pred_n in gold_n


def is_faithful(answer: str, contexts: list[str]) -> bool:
    """
    Faithful if answer content appears in retrieved context.
    """
    ans = normalize(answer)

    for ctx in contexts:
        if ans in normalize(ctx):
            return True

    return False
