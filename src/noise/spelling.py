import random

def apply_spelling_noise(text, p):
    chars = list(text)
    for i in range(len(chars)):
        if random.random() < p:
            op = random.choice(["delete", "swap"])
            if op == "delete":
                chars[i] = ""
            elif op == "swap" and i < len(chars) - 1:
                chars[i], chars[i+1] = chars[i+1], chars[i]
    return "".join(chars)
