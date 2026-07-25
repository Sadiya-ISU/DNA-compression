"""Round-trip tests for the DNACompress codec."""
from __future__ import annotations

import random

import pytest

from DNACompress import DNACompressDecoder, DNACompressEncoder


def _roundtrip(text: str) -> None:
    encoder = DNACompressEncoder()
    decoder = DNACompressDecoder()
    encoded = encoder.encode(text)
    decoded = decoder.decode(encoded)
    assert decoded == text


def test_dnacompress_roundtrip_random(random_dna_medium):
    _roundtrip(random_dna_medium)


def test_dnacompress_roundtrip_repetitive(repetitive_dna):
    _roundtrip(repetitive_dna)


def test_dnacompress_roundtrip_codon_conserved(codon_conserved_dna):
    _roundtrip(codon_conserved_dna)


@pytest.mark.parametrize("seed", list(range(10)))
def test_dnacompress_fuzz(seed):
    rng = random.Random(seed + 200)
    length = rng.randint(50, 3000)
    text = "".join(rng.choice("ACGT") for _ in range(length))
    _roundtrip(text)


def test_dnacompress_repetitive_compresses_well():
    """Regression test for the bug-#1 fix in `fixed.md`:
    after the add_text fix, 'ACGTACGT'*500 should compress to far less than
    the input size (the bug used to inflate it to 200%)."""
    text = "ACGTACGT" * 500  # 4 000 bp, period 8
    encoder = DNACompressEncoder()
    decoder = DNACompressDecoder()
    encoded = encoder.encode(text)
    assert decoder.decode(encoded) == text
    assert len(encoded) < len(text) // 4, (
        f"DNACompress regressed: 8-cycle periodic input compressed to {len(encoded)} B "
        f"(input {len(text)} B); expected substantial compression."
    )
