# fuzzy_sugeno.py (first-order Sugeno)
from typing import Tuple
import numpy as np
import pandas as pd

def sugeno_first_order_scalar(prob: float,
                              p1: float = -0.5, q1: float = 0.4,
                              p2: float =  0.7, q2: float = 0.2) -> float:
    """
    Sugeno first-order untuk satu nilai prob.
    p1,q1: koefisien consequen aturan 'low'
    p2,q2: koefisien consequen aturan 'high'
    """
    low = max(0.0, 1.0 - prob)
    high = max(0.0, prob)

    a_low  = p1 * prob + q1
    a_high = p2 * prob + q2

    denom = low + high
    if denom == 0:
        return 0.5
    return (low * a_low + high * a_high) / denom


def sugeno_first_order_vectorized(probs: np.ndarray,
                                  p1: float = -0.5, q1: float = 0.4,
                                  p2: float =  0.7, q2: float = 0.2) -> np.ndarray:
    """
    Vektorisasi: input array 1D probs -> output array y
    """
    probs = np.asarray(probs, dtype=float)
    lows = np.clip(1.0 - probs, 0.0, 1.0)
    highs = np.clip(probs, 0.0, 1.0)

    a_lows = p1 * probs + q1
    a_highs = p2 * probs + q2

    numer = lows * a_lows + highs * a_highs
    denom = lows + highs
    # hindari pembagian 0
    denom_safe = np.where(denom == 0, 1.0, denom)
    y = numer / denom_safe
    # default value ketika denom==0 -> 0.5
    y = np.where(denom == 0, 0.5, y)
    return y


def apply_to_dataframe(df: pd.DataFrame, prob_col: str = "Prob",
                       out_col: str = "Sugeno",
                       p1: float = -0.5, q1: float = 0.4,
                       p2: float =  0.7, q2: float = 0.2) -> pd.DataFrame:
    """
    Tambahkan kolom Sugeno ke dataframe (v1: memerlukan kolom Prob).
    """
    probs = df[prob_col].to_numpy()
    df[out_col] = sugeno_first_order_vectorized(probs, p1, q1, p2, q2)
    return df

