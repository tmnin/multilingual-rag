import random

SUBS = {
    "a": ["@", "4"],
    "e": ["3"],
    "i": ["1", "!"],
    "o": ["0"],
    "s": ["$", "5"],
    "l": ["1"]
}

def apply_obfuscation_noise(text, p):
    out = []
    for ch in text:
        if ch.lower() in SUBS and random.random() < p:
            out.append(random.choice(SUBS[ch.lower()]))
        else:
            out.append(ch)
    return "".join(out)
