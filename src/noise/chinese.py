import random

def inject_chinese_noise(text, prob=0.1):
    out = []
    for ch in text:
        # Skip punctuation / spaces
        if ch.strip() == "":
            out.append(ch)
            continue

        if random.random() < prob:
            # drop or replace character
            if random.random() < 0.5:
                continue
            else:
                out.append("□")  # placeholder symbol
        else:
            out.append(ch)

    return "".join(out)
