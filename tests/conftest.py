"""Shared pytest fixtures for the DNA compression test suite."""
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import List

import pytest

# Make the project root importable so tests can `import VCSDplus` etc.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import SequenceRecord, generate_dataset  # noqa: E402

DNA_ALPHABET = "ACGT"


def _random_dna(length: int, seed: int) -> str:
    """Reproducible random DNA string."""
    generator = random.Random(seed)
    return "".join(generator.choice(DNA_ALPHABET) for _ in range(length))


@pytest.fixture
def random_dna_short() -> str:
    return _random_dna(120, seed=1)


@pytest.fixture
def random_dna_medium() -> str:
    return _random_dna(2400, seed=2)


@pytest.fixture
def repetitive_dna() -> str:
    """4000 bp of a fixed 20-base pattern (deliberately not codon-aligned)."""
    pattern = "ACGTACGTAAACCCGGGTTT"  # 20 bases — interesting mix
    return (pattern * 200)[:4000]


@pytest.fixture
def codon_conserved_dna() -> str:
    """Use the project's own codon-conserved generator for realism."""
    records = generate_dataset(
        num_sequences=1, avg_length=1500, dataset_type="codon_conserved", seed=3
    )
    return records[0][1]


@pytest.fixture
def fasta_records_small() -> List[SequenceRecord]:
    return generate_dataset(
        num_sequences=5, avg_length=400, dataset_type="mixed", seed=42
    )


@pytest.fixture
def fasta_records_repetitive() -> List[SequenceRecord]:
    return generate_dataset(
        num_sequences=5, avg_length=400, dataset_type="repetitive", seed=43
    )
