"""Round-trip tests for the standard-compressor baselines."""
from __future__ import annotations

import random

import pytest

from baselines import (
    Bz2Decoder, Bz2Encoder,
    GzipDecoder, GzipEncoder,
    LzmaDecoder, LzmaEncoder,
    TwoBitDecoder, TwoBitEncoder,
    ZlibDecoder, ZlibEncoder,
)

CODEC_PAIRS = [
    (GzipEncoder, GzipDecoder, "gzip"),
    (LzmaEncoder, LzmaDecoder, "lzma"),
    (Bz2Encoder, Bz2Decoder, "bz2"),
    (ZlibEncoder, ZlibDecoder, "zlib"),
    (TwoBitEncoder, TwoBitDecoder, "2bit"),
]


@pytest.mark.parametrize("encoder_cls,decoder_cls,name", CODEC_PAIRS)
def test_baseline_roundtrip_random(random_dna_medium, encoder_cls, decoder_cls, name):
    encoded = encoder_cls().encode(random_dna_medium)
    assert decoder_cls().decode(encoded) == random_dna_medium, f"{name} failed round-trip"


@pytest.mark.parametrize("encoder_cls,decoder_cls,name", CODEC_PAIRS)
def test_baseline_roundtrip_repetitive(repetitive_dna, encoder_cls, decoder_cls, name):
    encoded = encoder_cls().encode(repetitive_dna)
    assert decoder_cls().decode(encoded) == repetitive_dna, f"{name} failed round-trip"


@pytest.mark.parametrize("encoder_cls,decoder_cls,name", CODEC_PAIRS)
def test_baseline_records_input(fasta_records_small, encoder_cls, decoder_cls, name):
    encoded = encoder_cls().encode(fasta_records_small)
    decoded = decoder_cls().decode(encoded)
    expected = "".join(seq for _, seq in fasta_records_small)
    assert decoded == expected, f"{name} failed records round-trip"


@pytest.mark.parametrize("seed", list(range(8)))
def test_twobit_pack_size(seed):
    """2-bit pack should be exactly ceil(N/4) + 4 bytes for length-N input."""
    rng = random.Random(seed + 500)
    length = rng.randint(0, 1000)
    text = "".join(rng.choice("ACGT") for _ in range(length))
    encoded = TwoBitEncoder().encode(text)
    expected_size = 4 + (length + 3) // 4
    assert len(encoded) == expected_size
    assert TwoBitDecoder().decode(encoded) == text


def test_twobit_information_theoretic_floor(repetitive_dna):
    """The TwoBit codec's bits-per-base must be exactly 2 (excluding the 4-byte
    length prefix) — this is the i.i.d. lower bound."""
    encoded = TwoBitEncoder().encode(repetitive_dna)
    body_bytes = len(encoded) - 4
    body_bits_per_base = body_bytes * 8 / len(repetitive_dna)
    assert abs(body_bits_per_base - 2.0) < 0.05  # padding rounds up by < 1 byte
