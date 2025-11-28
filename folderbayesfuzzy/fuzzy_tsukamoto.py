# fuzzy_tsukamoto.py

def tsukamoto_decision(prob: float) -> float:
    """
    Tsukamoto: konsekuen (z) monotonic.
    pakai dua himpunan:
    - low  → z_low  = 0.2
    - high → z_high = 0.8
    """

    # derajat low & high dari probabilitas
    low = max(0.0, 1 - prob)  # makin kecil prob, makin besar low
    high = max(0.0, prob)     # makin besar prob, makin besar high

    z_low = 0.2   # misal: turunkan sedikit
    z_high = 0.8  # misal: turunkan besar

    num = low * z_low + high * z_high
    den = low + high

    if den == 0:
        return 0.5

    return num / den