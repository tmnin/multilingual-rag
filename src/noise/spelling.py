import random

def inject_spelling_noise(text, prob=0.1):
    chars = list(text)
    for i in range(len(chars)):
        if random.random() < prob:
            op = random.choice(["delete", "swap"])
            if op == "delete":
                chars[i] = ""
            elif op == "swap" and i < len(chars) - 1:
                chars[i], chars[i+1] = chars[i+1], chars[i]
    return "".join(chars)
