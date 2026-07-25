"""Shannon entropy and k-th order conditional entropy for DNA strings.

Used to report empirical lower bounds on the achievable bits-per-symbol for
each input dataset — a context any compressor's reported bpb should be read
against.

Definitions:
- shannon_entropy(text):  H_0 = -sum_x p(x) log2 p(x).
  For DNA over {A,C,G,T} this is bounded above by 2 bits/symbol.
- kth_order_entropy(text, k):  H_k = sum_c p(c) * (-sum_x p(x|c) log2 p(x|c))
  where c ranges over the empirical k-grams immediately preceding each
  symbol.  H_k is non-increasing in k, and equals the i.i.d. lower bound
  achievable by any predictor that conditions on at most k preceding
  symbols.

Edge cases:
- Empty input returns 0.0.
- For text shorter than k, the function falls back to H_0.
- Bases outside ACGT are NOT filtered; pass already-normalized text.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, List, Optional


def shannon_entropy(text: str) -> float:
    """Order-0 (per-symbol) Shannon entropy in bits/symbol."""
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)


def kth_order_entropy(text: str, k: int) -> float:
    """k-th order empirical conditional entropy in bits/symbol.

    For k <= 0, equivalent to shannon_entropy(text).
    For len(text) <= k, falls back to shannon_entropy(text).
    """
    if k <= 0 or len(text) <= k:
        return shannon_entropy(text)
    contexts: dict[str, Counter] = {}
    total_followers = 0
    for i in range(k, len(text)):
        context = text[i - k : i]
        symbol = text[i]
        contexts.setdefault(context, Counter())[symbol] += 1
        total_followers += 1
    if total_followers == 0:
        return 0.0
    weighted_entropy = 0.0
    for symbol_counts in contexts.values():
        local_total = sum(symbol_counts.values())
        local_h = -sum(
            (c / local_total) * math.log2(c / local_total)
            for c in symbol_counts.values() if c > 0
        )
        weighted_entropy += (local_total / total_followers) * local_h
    return weighted_entropy


def base_composition(text: str) -> dict[str, float]:
    """Return per-base frequencies as a {base: probability} dict."""
    if not text:
        return {}
    counts = Counter(text)
    total = len(text)
    return {base: count / total for base, count in counts.items()}


def entropy_profile(text: str, ks: Iterable[int] = (0, 1, 2, 3, 4, 5, 6)) -> List[tuple[int, float]]:
    """Compute (k, H_k) pairs over a range of orders. Useful for plotting."""
    return [(k, kth_order_entropy(text, k) if k > 0 else shannon_entropy(text)) for k in ks]


def gc_content(text: str) -> float:
    """Fraction of bases that are G or C. NaN-equivalent (0.0) on empty input."""
    if not text:
        return 0.0
    counts = Counter(text)
    return (counts.get("G", 0) + counts.get("C", 0)) / len(text)


def report(text: str, label: Optional[str] = None) -> str:
    """Human-readable entropy report for one input."""
    lines = []
    if label:
        lines.append(f"=== {label} ===")
    lines.append(f"length:    {len(text)} bp")
    lines.append(f"GC%:       {gc_content(text) * 100:.2f}%")
    composition = base_composition(text)
    composition_str = ", ".join(
        f"{base}={composition.get(base, 0.0):.4f}" for base in "ACGT"
    )
    lines.append(f"composition: {composition_str}")
    profile = entropy_profile(text, ks=(0, 1, 2, 3, 4, 5))
    for k, h in profile:
        lines.append(f"  H_{k}: {h:.4f} bits/symbol  (lower bound at order {k})")
    return "\n".join(lines)


__all__ = [
    "shannon_entropy",
    "kth_order_entropy",
    "base_composition",
    "entropy_profile",
    "gc_content",
    "report",
]
