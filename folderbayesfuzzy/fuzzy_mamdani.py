def mamdani_decision(prob: float) -> float:
    """
    Input: probabilitas overload (0-1)
    Output: nilai adjustment (0-1),
            misal 0.7 = turunkan daya cukup besar.
    """

    # Membership Risk (low, medium, high)
    low = max(0.0, (0.5 - prob) / 0.5)          # turun dari 1 ke 0
    med = max(0.0, 1 - abs(prob - 0.5) / 0.25)  # segitiga di sekitar 0.5
    high = max(0.0, (prob - 0.5) / 0.5)         # naik dari 0 ke 1

    # Aturan:
    # IF risk low   THEN adjustment kecil (0.2)
    # IF risk med   THEN adjustment sedang (0.5)
    # IF risk high  THEN adjustment besar (0.8)

    num = low * 0.2 + med * 0.5 + high * 0.8
    den = low + med + high

    if den == 0:
        return 0.5  # netral kalau nggak ada membership

    return num / den