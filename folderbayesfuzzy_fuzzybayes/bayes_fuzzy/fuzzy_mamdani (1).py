import numpy as np

# =============================
# INPUT 1: RISK (0–1)
# =============================

def risk_low(x):
    if x <= 0.2: return 1.0
    if 0.2 < x < 0.4: return (0.4 - x) / 0.2
    return 0.0

def risk_med(x):
    if 0.2 < x < 0.5: return (x - 0.2) / 0.3
    if 0.5 <= x < 0.8: return (0.8 - x) / 0.3
    if x == 0.5: return 1.0
    return 0.0

def risk_high(x):
    if x <= 0.6: return 0.0
    if 0.6 < x < 0.8: return (x - 0.6) / 0.2
    return 1.0


# =============================
# INPUT 2: DEVIATION (kW)
# =============================

def dev_neg(d):
    # dev <= -40 kuat, naik ke 0 pada -10
    if d <= -40: return 1.0
    if -40 < d < -10: return (-10 - d) / 30
    return 0.0

def dev_neu(d):
    # dev -40 → 1 → +20
    if -40 < d < 0: return (d + 40) / 40
    if 0 <= d < 20: return (20 - d) / 20
    if d == 0: return 1.0
    return 0.0

def dev_pos(d):
    # dev >= +20 kuat
    if d <= 10: return 0.0
    if 10 < d < 30: return (d - 10) / 20
    return 1.0


# =============================
# OUTPUT MEMBERSHIP
# =============================

def out_small(z):
    if z <= 0: return 1.0
    if 0 < z < 0.4: return (0.4 - z) / 0.4
    return 0.0

def out_medium(z):
    if 0.2 < z < 0.5: return (z - 0.2) / 0.3
    if 0.5 <= z < 0.8: return (0.8 - z) / 0.3
    if z == 0.5: return 1.0
    return 0.0

def out_big(z):
    if z <= 0.6: return 0.0
    if 0.6 < z < 0.9: return (z - 0.6) / 0.3
    return 1.0


# =============================
# 2-INPUT MAMDANI ENGINE
# =============================

def mamdani_decision(prob, dev):

    # fuzzify risk
    r_low  = risk_low(prob)
    r_med  = risk_med(prob)
    r_high = risk_high(prob)

    # fuzzify deviation
    d_neg = dev_neg(dev)
    d_neu = dev_neu(dev)
    d_pos = dev_pos(dev)

    # RULES
    rules = []

    # Dev NEGATIVE
    rules.append(("small", min(r_low,  d_neg)))
    rules.append(("small", min(r_med,  d_neg)))
    rules.append(("medium",min(r_high, d_neg)))

    # Dev NEUTRAL
    rules.append(("small", min(r_low,  d_neu)))
    rules.append(("medium",min(r_med,  d_neu)))
    rules.append(("big",   min(r_high, d_neu)))

    # Dev POSITIVE
    rules.append(("medium",min(r_low,  d_pos)))
    rules.append(("big",   min(r_med,  d_pos)))
    rules.append(("big",   min(r_high, d_pos)))

    # Defuzzification
    z_range = np.linspace(0, 1, 1001)
    num = 0.0
    den = 0.0

    for z in z_range:
        mu = 0.0
        for (label, strength) in rules:
            if strength == 0:
                continue
            if label == "small":
                mu = max(mu, min(strength, out_small(z)))
            elif label == "medium":
                mu = max(mu, min(strength, out_medium(z)))
            elif label == "big":
                mu = max(mu, min(strength, out_big(z)))

        num += z * mu
        den += mu

    return 0.5 if den == 0 else num / den
