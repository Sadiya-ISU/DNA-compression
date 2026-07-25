"""Shared utilities for the VCSD+ compression project.

This module provides DNA sequence helpers, light-weight bitstream support,
FASTA I/O, simple dataset generation, and result serialization utilities.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union, Optional, Any
import csv
import json
import os
import random
import sys
import time

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None

DNA_ALPHABET = "ACGT"
COMPLEMENT_TABLE = str.maketrans("ACGTacgt", "TGCAtgca")

SequenceRecord = Tuple[str, str]


def _normalize_sequence(sequence: str) -> str:
    sequence = sequence.strip().upper()
    invalid = set(sequence) - set(DNA_ALPHABET)
    if invalid:
        raise ValueError(f"Sequence contains invalid DNA bases: {sorted(invalid)}")
    return sequence


def reverse_complement(sequence: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    return sequence.translate(COMPLEMENT_TABLE)[::-1].upper()


def hamming_distance(left: str, right: str) -> int:
    """Return the Hamming distance between two equal-length strings."""
    if len(left) != len(right):
        raise ValueError("Hamming distance requires sequences of equal length")
    return sum(1 for a, b in zip(left, right) if a != b)


def codon_chunks(sequence: str) -> Tuple[List[str], str]:
    """Split a DNA sequence into codons and a trailing suffix."""
    sequence = _normalize_sequence(sequence)
    codons = [sequence[index : index + 3] for index in range(0, len(sequence) - len(sequence) % 3, 3)]
    suffix = sequence[len(codons) * 3 :]
    return codons, suffix


def count_codons(sequence: str) -> int:
    """Count the number of complete codons in a sequence."""
    return len(sequence) // 3


def combine_sequences(sequences: Iterable[str]) -> str:
    """Concatenate multiple sequences into one string."""
    return "".join(sequence.strip().upper() for sequence in sequences)


def random_sequence(length: int, seed: Optional[int] = None) -> str:
    """Generate a random DNA sequence."""
    generator = random.Random(seed)
    return "".join(generator.choice(DNA_ALPHABET) for _ in range(length))


def repetitive_sequence(length: int, pattern_length: int, seed: Optional[int] = None) -> str:
    """Generate a repetitive DNA sequence by repeating a random pattern."""
    if pattern_length <= 0:
        raise ValueError("pattern_length must be positive")
    generator = random.Random(seed)
    pattern = "".join(generator.choice(DNA_ALPHABET) for _ in range(pattern_length))
    repeats = (length // pattern_length) + 1
    return (pattern * repeats)[:length]


def generate_dataset(
    num_sequences: int,
    avg_length: int,
    dataset_type: str = "mixed",
    seed: Optional[int] = None,
) -> List[SequenceRecord]:
    """Generate a synthetic DNA dataset.

    Returns a list of (name, sequence) records.
    """
    if num_sequences <= 0:
        raise ValueError("num_sequences must be positive")
    if avg_length <= 0:
        raise ValueError("avg_length must be positive")

    generator = random.Random(seed)
    records: List[SequenceRecord] = []

    def _codon_conserved_sequence(length: int) -> str:
        codon_count = max(1, length // 3)
        motif_size = 10
        motif = [
            "".join(generator.choice(DNA_ALPHABET) for _ in range(3))
            for _ in range(motif_size)
        ]
        codons: List[str] = []
        for index in range(codon_count):
            base_codon = motif[index % motif_size]
            codon_chars = list(base_codon)
            # Moderate variation: mutate roughly 1 in 5 codons at one base.
            if generator.random() < 0.2:
                mutate_pos = generator.randint(0, 2)
                choices = [base for base in DNA_ALPHABET if base != codon_chars[mutate_pos]]
                codon_chars[mutate_pos] = generator.choice(choices)
            codons.append("".join(codon_chars))
        sequence = "".join(codons)
        trailing = length - len(sequence)
        if trailing > 0:
            sequence += "".join(generator.choice(DNA_ALPHABET) for _ in range(trailing))
        return sequence

    for index in range(num_sequences):
        jitter = generator.randint(-avg_length // 5, avg_length // 5)
        length = max(1, avg_length + jitter)
        if dataset_type == "random":
            sequence = random_sequence(length, seed=generator.randint(0, 10**9))
        elif dataset_type == "repetitive":
            pattern_length = max(3, min(18, length // 10 or 3))
            sequence = repetitive_sequence(length, pattern_length, seed=generator.randint(0, 10**9))
        elif dataset_type == "codon_conserved":
            sequence = _codon_conserved_sequence(length)
        else:
            if index % 2 == 0:
                sequence = repetitive_sequence(length, max(3, min(15, length // 12 or 3)), seed=generator.randint(0, 10**9))
            else:
                sequence = random_sequence(length, seed=generator.randint(0, 10**9))
        records.append((f"sequence_{index + 1}", sequence))
    return records


def save_sequences_to_fasta(sequences: Iterable[Union[str, SequenceRecord]], filename: Union[str, os.PathLike[str]]) -> None:
    """Write sequences to a FASTA file."""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index, entry in enumerate(sequences, start=1):
            if isinstance(entry, tuple):
                name, sequence = entry
            else:
                name, sequence = f"sequence_{index}", entry
            sequence = _normalize_sequence(sequence)
            handle.write(f">{name}\n")
            for offset in range(0, len(sequence), 80):
                handle.write(sequence[offset : offset + 80] + "\n")


def load_sequences_from_fasta(filename: Union[str, os.PathLike[str]]) -> List[SequenceRecord]:
    """Load sequences from a FASTA file."""
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(path)
    records: List[SequenceRecord] = []
    name: Optional[str] = None
    chunks: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append((name, _normalize_sequence("".join(chunks))))
                name = line[1:].strip() or f"sequence_{len(records) + 1}"
                chunks = []
            else:
                chunks.append(line)
    if name is not None:
        records.append((name, _normalize_sequence("".join(chunks))))
    return records


def save_results_csv(results: Sequence[dict], filename: Union[str, os.PathLike[str]]) -> None:
    """Save result dictionaries to a CSV file."""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def save_results_json(results: Any, filename: Union[str, os.PathLike[str]]) -> None:
    """Save any JSON-serializable object to a file."""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)


def validate_lossless(original: str, decoded: str) -> bool:
    """Return True if the two sequences are identical."""
    return _normalize_sequence(original) == _normalize_sequence(decoded)


def get_memory_usage() -> float:
    """Return the current process memory usage in MB."""
    if psutil is not None:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform.startswith("darwin"):
            return usage / (1024 * 1024)
        return usage / 1024
    except Exception:
        return 0.0


def format_bytes(size: int) -> str:
    """Format a byte count for readability."""
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{size} B"


@dataclass
class CompressionMetrics:
    """Aggregate compression metrics for a single algorithm/dataset pair.

    `original_size` is in bytes of input (here equal to `original_bases` because
    inputs are 1 ASCII byte per base). `compression_ratio` is therefore measured
    against the unpacked-ASCII baseline, not the 2-bits-per-base information
    lower bound; use `bits_per_base` for that comparison.
    """

    algorithm: str = ""
    dataset: str = ""
    original_size: int = 0
    compressed_size: int = 0
    encode_time: float = 0.0
    decode_time: float = 0.0
    peak_memory: float = 0.0
    dictionary_size: float = 0.0
    original_bases: int = 0
    lossless: bool = False

    def calculate_compression_ratio(self) -> float:
        if self.original_size == 0:
            return 0.0
        return (self.compressed_size / self.original_size) * 100.0

    def calculate_bits_per_base(self) -> float:
        if self.original_bases == 0:
            return 0.0
        return (self.compressed_size * 8.0) / self.original_bases

    def calculate_throughput(self, is_encoding: bool = True) -> float:
        duration = self.encode_time if is_encoding else self.decode_time
        if duration <= 0 or self.original_bases == 0:
            return 0.0
        return self.original_bases / duration

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload.update(
            {
                "compression_ratio_%": round(self.calculate_compression_ratio(), 4),
                "bits_per_base": round(self.calculate_bits_per_base(), 4),
                "encode_throughput_bases_per_sec": round(self.calculate_throughput(True), 4),
                "decode_throughput_bases_per_sec": round(self.calculate_throughput(False), 4),
            }
        )
        return payload

    def __str__(self) -> str:
        return (
            f"{self.algorithm} on {self.dataset}: "
            f"ratio={self.calculate_compression_ratio():.2f}%, "
            f"bpb={self.calculate_bits_per_base():.3f}, "
            f"encode={self.encode_time:.4f}s, decode={self.decode_time:.4f}s, "
            f"memory={self.peak_memory:.2f} MB, lossless={'YES' if self.lossless else 'NO'}"
        )


def current_timestamp() -> str:
    """Return an ISO-like timestamp without timezone information."""
    return time.strftime("%Y-%m-%dT%H:%M:%S")
