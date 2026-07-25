"""Round-trip tests for the LZSS codec."""
from __future__ import annotations

import random

import pytest

from LZSS import LZSSDecoder, LZSSEncoder


def _roundtrip(text: str, **encoder_kwargs) -> None:
    encoder = LZSSEncoder(**encoder_kwargs)
    decoder = LZSSDecoder()
    encoded = encoder.encode(text)
    decoded = decoder.decode(encoded)
    assert decoded == text


def test_lzss_roundtrip_random(random_dna_medium):
    _roundtrip(random_dna_medium)


def test_lzss_roundtrip_repetitive(repetitive_dna):
    _roundtrip(repetitive_dna)


def test_lzss_roundtrip_codon_conserved(codon_conserved_dna):
    _roundtrip(codon_conserved_dna)


@pytest.mark.parametrize("seed", list(range(10)))
def test_lzss_fuzz(seed):
    rng = random.Random(seed + 400)
    length = rng.randint(50, 3000)
    text = "".join(rng.choice("ACGT") for _ in range(length))
    _roundtrip(text)


@pytest.mark.parametrize("window_size,lookahead", [(256, 16), (1024, 32), (4096, 64)])
def test_lzss_window_sizes(repetitive_dna, window_size, lookahead):
    _roundtrip(repetitive_dna, window_size=window_size, lookahead_buffer_size=lookahead)
