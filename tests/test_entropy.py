"""Tests for the entropy module."""
from __future__ import annotations

import math

import pytest

from entropy import (
    base_composition,
    entropy_profile,
    gc_content,
    kth_order_entropy,
    shannon_entropy,
)


def test_shannon_entropy_uniform_dna():
    """Equal counts of A/C/G/T -> exactly 2 bits/symbol."""
    text = "ACGT" * 250
    assert abs(shannon_entropy(text) - 2.0) < 1e-9


def test_shannon_entropy_homopolymer():
    """A single distinct symbol -> 0 bits/symbol."""
    assert shannon_entropy("AAAAA") == 0.0


def test_shannon_entropy_empty():
    assert shannon_entropy("") == 0.0


def test_shannon_entropy_two_symbols_equal():
    """50/50 mix of two symbols -> 1 bit/symbol."""
    assert abs(shannon_entropy("AC" * 100) - 1.0) < 1e-9


def test_kth_order_entropy_periodic_drops():
    """A purely periodic string with period 4 should have H_3 close to 0:
    once you know three preceding bases of 'ACGT' you can predict the next."""
    text = "ACGT" * 1000
    h0 = shannon_entropy(text)
    h3 = kth_order_entropy(text, 3)
    assert h0 == pytest.approx(2.0, abs=1e-9)
    assert h3 < 0.01, f"H_3 should be ~0 for period-4 string, got {h3}"


def test_kth_order_entropy_random_does_not_drop_much():
    """Truly random DNA should have H_k ~= H_0 for all small k."""
    import random
    rng = random.Random(0)
    text = "".join(rng.choice("ACGT") for _ in range(20_000))
    h0 = shannon_entropy(text)
    for k in (1, 2, 3, 4):
        h_k = kth_order_entropy(text, k)
        # Random strings of finite length give a small finite-sample shrinkage
        # relative to H_0, but it should not exceed 0.05 bits at this length.
        assert h0 - h_k < 0.05, f"H_{k}={h_k} unexpectedly << H_0={h0}"


def test_kth_order_entropy_falls_back_for_short_input():
    """k larger than len(text) falls back to H_0."""
    text = "ACGT"
    assert kth_order_entropy(text, 100) == shannon_entropy(text)


def test_kth_order_entropy_zero_k_equals_shannon():
    text = "ACCGGT" * 50
    assert kth_order_entropy(text, 0) == shannon_entropy(text)


def test_base_composition_sums_to_one():
    text = "ACGTACGTAA"
    composition = base_composition(text)
    assert math.isclose(sum(composition.values()), 1.0)


def test_gc_content():
    assert gc_content("AAAA") == 0.0
    assert gc_content("CGCG") == 1.0
    assert abs(gc_content("ACGT") - 0.5) < 1e-9
    assert gc_content("") == 0.0


def test_entropy_profile_monotone_non_increasing():
    """H_k is mathematically non-increasing in k. Tolerate finite-sample noise."""
    import random
    rng = random.Random(7)
    text = "".join(rng.choice("ACGT") for _ in range(5000))
    profile = entropy_profile(text, ks=(0, 1, 2, 3, 4, 5))
    values = [h for _, h in profile]
    for prev, curr in zip(values, values[1:]):
        # Allow small finite-sample bumps up to 0.05 bits.
        assert curr <= prev + 0.05, f"H_k profile not monotone: {values}"
