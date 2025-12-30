import random
from noise.spelling import apply_spelling_noise
from noise.codeswitch import apply_codeswitch_noise
from noise.obfuscation import apply_obfuscation_noise
from noise.chinese import apply_chinese_noise


def apply_noise(text: str, language: str, noise_p: float) -> str:
    if noise_p == 0.0:
        return text
    
    if language == "zh":
        return apply_chinese_noise(text, noise_p)
    
    if random.random() < noise_p:
        text = apply_spelling_noise(text, noise_p)

    if random.random() < noise_p:
        text = apply_codeswitch_noise(text, noise_p)

    if random.random() < noise_p:
        text = apply_obfuscation_noise(text, noise_p)

    return text
