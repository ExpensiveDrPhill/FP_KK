import numpy as np

# -----------------------
# Helper membership functions (manual version of trapmf, trimf, sigmf)
# -----------------------

def trapmf(x, a, b, c, d):
    if x <= a: return 0
    if a <= x <= b: return (x - a) / (b - a)
    if b <= x <= c: return 1
    if c <= x <= d: return (d - x) / (d - c)
    return 0

def trimf(x, a, b, c):
    if x <= a or x >= c: return 0
    if a < x < b: return (x - a) / (b - a)
    if b <= x < c: return (c - x) / (c - b)
    return 0

def sigmf(x, b, c):
    # equivalent skfuzzy: fuzz.sigmf(x, b, c)
    return 1 / (1 + np.exp(-c * (x - b)))


# -----------------------
# Sugeno System
# -----------------------
class GridEarlyWarning_Sugeno:
    def __init__(self):
        pass

    # -----------------------------
    # Fuzzification with SAME MF as Mamdani
    # -----------------------------
    def fuzzify(self, time, day, rain):
        # Time
        mu_morning   = trimf(time, 7, 9, 11)
        mu_afternoon = trapmf(time, 10, 12, 17, 19)
        mu_night     = trapmf(time, 0, 0, 6, 8)
        mu_evening   = trapmf(time, 18, 20, 24, 24)

        # Day
        mu_weekday = trapmf(day, 0, 0, 4, 5)
        mu_weekend = trapmf(day, 4, 5, 6, 6)

        # Weather
        mu_clear = sigmf(rain, 0.5, -5)
        mu_heavy = sigmf(rain, 0.5, 5)

        return {
            "morning": mu_morning,
            "afternoon": mu_afternoon,
            "night": mu_night,
            "evening": mu_evening,
            "weekday": mu_weekday,
            "weekend": mu_weekend,
            "clear": mu_clear,
            "heavy": mu_heavy
        }

    # -----------------------------
    # Evaluate Sugeno
    # -----------------------------
    def evaluate(self, time, day, rain):
        f = self.fuzzify(time, day, rain)

        # ===== RULES (SAMA DENGAN MAMDANI) =====

        # High
        r1 = min(f["morning"], f["weekday"])
        r2 = min(f["morning"], f["heavy"])
        r3 = min(f["afternoon"], min(f["weekday"], f["heavy"]))
        high = max(r1, r2, r3)

        # Medium
        r4 = min(f["morning"], min(f["weekend"], f["clear"]))
        r5 = min(f["afternoon"], min(f["weekday"], f["clear"]))
        r6 = min(f["afternoon"], min(f["weekend"], f["heavy"]))
        med = max(r4, r5, r6)

        # Low
        r7 = f["night"]
        r8 = f["evening"]
        r9 = min(f["afternoon"], min(f["weekend"], f["clear"]))
        low = max(r7, r8, r9)

        # ============================
        # Sugeno OUTPUT CONSTANTS (0–100)
        # ============================
        z_low = 20
        z_med = 50
        z_high = 85

        # Weighted average
        numerator = low*z_low + med*z_med + high*z_high
        denominator = low + med + high

        if denominator == 0:
            return 0.0

        return numerator / denominator
