import random

SUBS = {
    "a": ["@", "4"],
    "e": ["3"],
    "i": ["1", "!"],
    "o": ["0"],
    "s": ["$", "5"],
    "l": ["1"]
}

def inject_obfuscation(text, prob=0.1):
    out = []
    for ch in text:
        if ch.lower() in SUBS and random.random() < prob:
            out.append(random.choice(SUBS[ch.lower()]))
        else:
            out.append(ch)
    return "".join(out)
