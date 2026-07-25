"""Round-trip tests for the LZ78-style codec."""
from __future__ import annotations

import random

import pytest

from LZ78_style import LZ78Decoder, LZ78Encoder


def _roundtrip(text: str) -> None:
    encoder = LZ78Encoder()
    decoder = LZ78Decoder()
    encoded = encoder.encode(text)
    decoded = decoder.decode(encoded)
    assert decoded == text


def test_lz78_roundtrip_random(random_dna_medium):
    _roundtrip(random_dna_medium)


def test_lz78_roundtrip_repetitive(repetitive_dna):
    _roundtrip(repetitive_dna)


def test_lz78_roundtrip_codon_conserved(codon_conserved_dna):
    _roundtrip(codon_conserved_dna)


@pytest.mark.parametrize("seed", list(range(10)))
def test_lz78_fuzz(seed):
    rng = random.Random(seed + 300)
    length = rng.randint(50, 3000)
    text = "".join(rng.choice("ACGT") for _ in range(length))
    _roundtrip(text)


def test_lz78_records_input(fasta_records_small):
    """LZ78 accepts both str and Sequence[SequenceRecord]; check the records path."""
    encoder = LZ78Encoder()
    decoder = LZ78Decoder()
    encoded = encoder.encode(fasta_records_small)
    decoded = decoder.decode(encoded)
    expected = "".join(seq for _, seq in fasta_records_small)
    assert decoded == expected
