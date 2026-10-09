"""Success-rate statistics. Wilson score interval [Wilson1927] (see plans/references.md)."""

import math


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n trials."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def summarize(successes: list[bool]) -> dict:
    k, n = sum(successes), len(successes)
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": k / n if n else 0.0, "ci95": [lo, hi]}
