"""Round-trip property tests for VCSD+ across all advertised modes."""
from __future__ import annotations

import random

import pytest

from VCSDplus import VCSDDecoder, VCSDEncoder
from utils import SequenceRecord


def _roundtrip_records(records, **encoder_kwargs):
    encoder = VCSDEncoder(**encoder_kwargs)
    decoder = VCSDDecoder()
    encoded = encoder.encode(records)
    decoded = decoder.decode(encoded)
    assert len(decoded) == len(records), "record count must round-trip"
    decoded_text = "".join(seq for _, seq in decoded)
    original_text = "".join(seq for _, seq in records)
    assert decoded_text == original_text, "concatenated sequence text must round-trip"


@pytest.mark.parametrize(
    "use_approximate,use_reverse_complement,use_token_entropy",
    [
        (False, False, False),  # compact-exact
        (False, False, True),   # compact-exact + entropy
        (True, False, False),   # extended (approx)
        (False, True, False),   # extended (RC)
        (True, True, False),    # extended (approx + RC)
    ],
)
def test_vcsd_roundtrip_all_modes(
    fasta_records_small,
    use_approximate,
    use_reverse_complement,
    use_token_entropy,
):
    _roundtrip_records(
        fasta_records_small,
        use_approximate=use_approximate,
        use_reverse_complement=use_reverse_complement,
        use_token_entropy=use_token_entropy,
    )


@pytest.mark.parametrize("ordering", [False, True])
def test_vcsd_sequence_ordering_roundtrip(fasta_records_small, ordering):
    _roundtrip_records(fasta_records_small, sequence_ordering=ordering)


def test_vcsd_sequence_ordering_preserves_original_index(fasta_records_small):
    """With sequence_ordering=True, decoded records must come back in input order."""
    encoder = VCSDEncoder(sequence_ordering=True)
    decoder = VCSDDecoder()
    encoded = encoder.encode(fasta_records_small)
    decoded = decoder.decode(encoded)
    for i, (_, original_seq) in enumerate(fasta_records_small):
        assert decoded[i][1] == original_seq, f"record {i} not at original index after ordering"


def test_vcsd_string_input_roundtrip(random_dna_medium):
    encoder = VCSDEncoder()
    decoder = VCSDDecoder()
    encoded = encoder.encode(random_dna_medium)
    decoded = decoder.decode(encoded)
    assert "".join(seq for _, seq in decoded) == random_dna_medium


@pytest.mark.parametrize("seed", list(range(10)))
def test_vcsd_compact_roundtrip_fuzz(seed):
    """10 reproducible random inputs of varying length, compact-exact mode."""
    rng = random.Random(seed)
    length = rng.randint(50, 3000)
    text = "".join(rng.choice("ACGT") for _ in range(length))
    records: list[SequenceRecord] = [(f"seq_{seed}", text)]
    _roundtrip_records(records)


@pytest.mark.parametrize("seed", list(range(10)))
def test_vcsd_entropy_roundtrip_fuzz(seed):
    rng = random.Random(seed + 100)
    length = rng.randint(50, 3000)
    text = "".join(rng.choice("ACGT") for _ in range(length))
    records: list[SequenceRecord] = [(f"seq_{seed}", text)]
    _roundtrip_records(records, use_token_entropy=True)
