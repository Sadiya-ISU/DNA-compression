"""Unified benchmarking and comparative analysis.

Runs all four DNA-specific codecs (VCSD+ compact and entropy modes,
DNACompress, LZ78-style, LZSS) and the five general-purpose stdlib
baselines (gzip, lzma, bz2, raw zlib, 2-bit pack) across the synthetic
sample datasets and any real reference genomes present under
``sample_data/``, writing reports under ``Results/``.
"""
from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from LZ78_style import LZ78Decoder, LZ78Encoder
from LZSS import LZSSDecoder, LZSSEncoder
from DNACompress import DNACompressDecoder, DNACompressEncoder
from VCSDplus import VCSDDecoder, VCSDEncoder
from baselines import (
    Bz2Decoder, Bz2Encoder,
    GzipDecoder, GzipEncoder,
    LzmaDecoder, LzmaEncoder,
    TwoBitDecoder, TwoBitEncoder,
    ZlibDecoder, ZlibEncoder,
)
from entropy import report as entropy_report
from plots import main as run_plots
from utils import (
    CompressionMetrics,
    SequenceRecord,
    current_timestamp,
    generate_dataset,
    load_sequences_from_fasta,
    save_results_csv,
    save_results_json,
    save_sequences_to_fasta,
    validate_lossless,
)

PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"
RESULTS_DIR = PROJECT_ROOT / "Results"

DEFAULT_DATASETS = {
    "small_test.fasta": {"num_sequences": 5, "avg_length": 500, "dataset_type": "mixed"},
    "medium_test.fasta": {"num_sequences": 10, "avg_length": 800, "dataset_type": "mixed"},
    "large_test.fasta": {"num_sequences": 20, "avg_length": 1000, "dataset_type": "mixed"},
    "repetitive_codon_test.fasta": {"num_sequences": 20, "avg_length": 1200, "dataset_type": "repetitive"},
    "codon_conserved_variation_test.fasta": {
        "num_sequences": 20,
        "avg_length": 1200,
        "dataset_type": "codon_conserved",
    },
}

# Real public-domain reference genomes; expected to be present in sample_data/
# after running fetch_real_genomes.py. The benchmark loads them if available
# but does not synthesize them.
REAL_GENOME_FILES = [
    "phiX174.fasta",
    "lambda_phage.fasta",
]


def _dataset_bases(records: Sequence[SequenceRecord]) -> int:
    return sum(len(sequence) for _, sequence in records)


def _dataset_text(records: Sequence[SequenceRecord]) -> str:
    return "".join(sequence for _, sequence in records)


def _memory_peak_mb(action) -> Tuple[float, object]:
    tracemalloc.start()
    result = action()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024), result


class CompressionBenchmark:
    """Benchmark orchestration for the three codecs."""

    def __init__(self) -> None:
        pass

    def run_all_algorithms(self, dataset: Sequence[SequenceRecord], dataset_name: str) -> List[CompressionMetrics]:
        results: List[CompressionMetrics] = []
        # Project codecs.
        results.append(self._run_vcsd(dataset, dataset_name))
        results.append(self._run_vcsd_entropy(dataset, dataset_name))
        results.append(self._run_dnacompress(dataset, dataset_name))
        results.append(self._run_lz78(dataset, dataset_name))
        results.append(self._run_lzss(dataset, dataset_name))
        # Standard general-purpose baselines (stdlib).
        results.append(self._run_twobit(dataset, dataset_name))
        results.append(self._run_gzip(dataset, dataset_name))
        results.append(self._run_zlib(dataset, dataset_name))
        results.append(self._run_bz2(dataset, dataset_name))
        results.append(self._run_lzma(dataset, dataset_name))
        return results

    def measure_encoding_time(self, encoder, data) -> Tuple[float, bytes]:
        start = time.perf_counter()
        encoded = encoder.encode(data)
        elapsed = time.perf_counter() - start
        return elapsed, encoded

    def measure_decoding_time(self, decoder, encoded) -> Tuple[float, object]:
        start = time.perf_counter()
        decoded = decoder.decode(encoded)
        elapsed = time.perf_counter() - start
        return elapsed, decoded

    def measure_memory_usage(self, encoder, decoder, data) -> float:
        def _action():
            encoded = encoder.encode(data)
            return decoder.decode(encoded)

        peak_mb, _ = _memory_peak_mb(_action)
        return peak_mb

    def validate_lossless(self, original, decoded) -> bool:
        if isinstance(original, str) and isinstance(decoded, str):
            return validate_lossless(original, decoded)
        if isinstance(original, Sequence) and isinstance(decoded, Sequence):
            original_text = "".join(sequence for _, sequence in original)
            if decoded and isinstance(decoded[0], tuple):
                decoded_text = "".join(sequence for _, sequence in decoded)
            else:
                decoded_text = "".join(decoded)  # type: ignore[arg-type]
            return validate_lossless(original_text, decoded_text)
        return False

    def _run_vcsd(self, dataset: Sequence[SequenceRecord], dataset_name: str) -> CompressionMetrics:
        encoder = VCSDEncoder(use_approximate=False, use_reverse_complement=False, sequence_ordering=True)
        decoder = VCSDDecoder()
        original_bases = _dataset_bases(dataset)
        original_size = original_bases

        encode_time, encoded = self.measure_encoding_time(encoder, dataset)
        decode_time, decoded = self.measure_decoding_time(decoder, encoded)
        peak_memory = self.measure_memory_usage(VCSDEncoder(use_approximate=False, use_reverse_complement=False, sequence_ordering=True), VCSDDecoder(), dataset)
        decoded_text = "".join(sequence for _, sequence in decoded)
        lossless = self.validate_lossless(dataset, decoded)
        # Total codons stored across all dictionary phrases, in KB at 1 byte per codon-id.
        dictionary_size = sum(len(phrase) for phrase in encoder.dictionary.phrases) / 1024.0

        return CompressionMetrics(
            algorithm="VCSDplus",
            dataset=dataset_name,
            original_size=original_size,
            compressed_size=len(encoded),
            encode_time=encode_time,
            decode_time=decode_time,
            peak_memory=peak_memory,
            dictionary_size=dictionary_size,
            original_bases=original_bases,
            lossless=lossless and bool(decoded_text),
        )

    def _run_lz78(self, dataset: Sequence[SequenceRecord], dataset_name: str) -> CompressionMetrics:
        encoder = LZ78Encoder()
        decoder = LZ78Decoder()
        text = _dataset_text(dataset)
        original_bases = len(text)
        original_size = original_bases

        encode_time, encoded = self.measure_encoding_time(encoder, text)
        decode_time, decoded = self.measure_decoding_time(decoder, encoded)
        peak_memory = self.measure_memory_usage(LZ78Encoder(), LZ78Decoder(), text)
        lossless = self.validate_lossless(text, decoded)
        # Total characters stored across all dictionary phrases, in KB.
        dictionary_size = sum(len(phrase) for phrase in encoder.dictionary.phrases) / 1024.0

        return CompressionMetrics(
            algorithm="LZ78_style",
            dataset=dataset_name,
            original_size=original_size,
            compressed_size=len(encoded),
            encode_time=encode_time,
            decode_time=decode_time,
            peak_memory=peak_memory,
            dictionary_size=dictionary_size,
            original_bases=original_bases,
            lossless=lossless,
        )

    def _run_vcsd_entropy(self, dataset: Sequence[SequenceRecord], dataset_name: str) -> CompressionMetrics:
        encoder = VCSDEncoder(
            use_approximate=False,
            use_reverse_complement=False,
            sequence_ordering=True,
            use_token_entropy=True,
        )
        decoder = VCSDDecoder()
        original_bases = _dataset_bases(dataset)
        original_size = original_bases

        encode_time, encoded = self.measure_encoding_time(encoder, dataset)
        decode_time, decoded = self.measure_decoding_time(decoder, encoded)
        peak_memory = self.measure_memory_usage(
            VCSDEncoder(
                use_approximate=False,
                use_reverse_complement=False,
                sequence_ordering=True,
                use_token_entropy=True,
            ),
            VCSDDecoder(),
            dataset,
        )
        decoded_text = "".join(sequence for _, sequence in decoded)
        lossless = self.validate_lossless(dataset, decoded)
        # Total codons stored across all dictionary phrases, in KB at 1 byte per codon-id.
        dictionary_size = sum(len(phrase) for phrase in encoder.dictionary.phrases) / 1024.0

        return CompressionMetrics(
            algorithm="VCSDplus_entropy",
            dataset=dataset_name,
            original_size=original_size,
            compressed_size=len(encoded),
            encode_time=encode_time,
            decode_time=decode_time,
            peak_memory=peak_memory,
            dictionary_size=dictionary_size,
            original_bases=original_bases,
            lossless=lossless and bool(decoded_text),
        )

    def _run_dnacompress(self, dataset: Sequence[SequenceRecord], dataset_name: str) -> CompressionMetrics:
        encoder = DNACompressEncoder()
        decoder = DNACompressDecoder()
        text = _dataset_text(dataset)
        original_bases = len(text)
        original_size = original_bases

        encode_time, encoded = self.measure_encoding_time(encoder, text)
        decode_time, decoded = self.measure_decoding_time(decoder, encoded)
        peak_memory = self.measure_memory_usage(DNACompressEncoder(), DNACompressDecoder(), text)
        lossless = self.validate_lossless(text, decoded)
        # Stored seed positions across forward + RC buckets, at 4 bytes per int, in KB.
        forward_entries = sum(len(positions) for positions in encoder.index.forward_positions.values())
        rc_entries = sum(len(positions) for positions in encoder.index.rc_positions.values())
        dictionary_size = (forward_entries + rc_entries) * 4 / 1024.0

        return CompressionMetrics(
            algorithm="DNACompress",
            dataset=dataset_name,
            original_size=original_size,
            compressed_size=len(encoded),
            encode_time=encode_time,
            decode_time=decode_time,
            peak_memory=peak_memory,
            dictionary_size=dictionary_size,
            original_bases=original_bases,
            lossless=lossless,
        )

    def _run_lzss(self, dataset: Sequence[SequenceRecord], dataset_name: str) -> CompressionMetrics:
        encoder = LZSSEncoder(window_size=1024, lookahead_buffer_size=32)
        decoder = LZSSDecoder()
        text = _dataset_text(dataset)
        original_bases = len(text)
        original_size = original_bases

        encode_time, encoded = self.measure_encoding_time(encoder, text)
        decode_time, decoded = self.measure_decoding_time(decoder, encoded)
        peak_memory = self.measure_memory_usage(LZSSEncoder(window_size=1024, lookahead_buffer_size=32), LZSSDecoder(), text)
        lossless = self.validate_lossless(text, decoded)
        # LZSS keeps no dictionary; it scans a fixed-size window of the input
        # itself. Report the upper-bound window footprint as a memory descriptor.
        dictionary_size = encoder.window.window_size / 1024.0

        return CompressionMetrics(
            algorithm="LZSS",
            dataset=dataset_name,
            original_size=original_size,
            compressed_size=len(encoded),
            encode_time=encode_time,
            decode_time=decode_time,
            peak_memory=peak_memory,
            dictionary_size=dictionary_size,
            original_bases=original_bases,
            lossless=lossless,
        )

    def _run_simple_codec(
        self,
        algorithm: str,
        encoder_factory,
        decoder_factory,
        dataset: Sequence[SequenceRecord],
        dataset_name: str,
    ) -> CompressionMetrics:
        encoder = encoder_factory()
        decoder = decoder_factory()
        text = _dataset_text(dataset)
        original_bases = len(text)
        original_size = original_bases

        encode_time, encoded = self.measure_encoding_time(encoder, text)
        decode_time, decoded = self.measure_decoding_time(decoder, encoded)
        peak_memory = self.measure_memory_usage(encoder_factory(), decoder_factory(), text)
        lossless = self.validate_lossless(text, decoded)
        # Stdlib baselines have no exposed in-memory dictionary; report 0.
        dictionary_size = 0.0

        return CompressionMetrics(
            algorithm=algorithm,
            dataset=dataset_name,
            original_size=original_size,
            compressed_size=len(encoded),
            encode_time=encode_time,
            decode_time=decode_time,
            peak_memory=peak_memory,
            dictionary_size=dictionary_size,
            original_bases=original_bases,
            lossless=lossless,
        )

    def _run_gzip(self, dataset, dataset_name):
        return self._run_simple_codec("gzip_lvl9", GzipEncoder, GzipDecoder, dataset, dataset_name)

    def _run_lzma(self, dataset, dataset_name):
        return self._run_simple_codec("lzma_default", LzmaEncoder, LzmaDecoder, dataset, dataset_name)

    def _run_bz2(self, dataset, dataset_name):
        return self._run_simple_codec("bz2_lvl9", Bz2Encoder, Bz2Decoder, dataset, dataset_name)

    def _run_zlib(self, dataset, dataset_name):
        return self._run_simple_codec("zlib_lvl9", ZlibEncoder, ZlibDecoder, dataset, dataset_name)

    def _run_twobit(self, dataset, dataset_name):
        return self._run_simple_codec("2bit_pack", TwoBitEncoder, TwoBitDecoder, dataset, dataset_name)


class ResultReporter:
    """Generate benchmark reports in multiple formats."""

    def generate_text_report(self, results: Sequence[CompressionMetrics], output_path: Path) -> None:
        lines = ["COMPRESSION RATIO ANALYSIS", "=========================", ""]
        for metrics in results:
            lines.append(f"Algorithm: {metrics.algorithm}")
            lines.append(f"Dataset: {metrics.dataset}")
            lines.append(f"  Original Size:      {metrics.original_size} bytes")
            lines.append(f"  Compressed Size:    {metrics.compressed_size} bytes")
            lines.append(f"  Compression Ratio:  {metrics.calculate_compression_ratio():.2f}%")
            lines.append(f"  Bits per Base:      {metrics.calculate_bits_per_base():.3f}")
            lines.append(f"  Encode Time:        {metrics.encode_time:.4f} s")
            lines.append(f"  Decode Time:        {metrics.decode_time:.4f} s")
            lines.append(f"  Peak Memory:        {metrics.peak_memory:.2f} MB")
            lines.append(f"  Dictionary Size:    {metrics.dictionary_size:.2f} KB")
            lines.append(f"  Lossless:           {'YES' if metrics.lossless else 'NO'}")
            lines.append("")
        lines.append("Summary Table:")
        for metrics in results:
            lines.append(
                f"{metrics.algorithm:12s} | {metrics.dataset:16s} | "
                f"{metrics.calculate_compression_ratio():7.2f}% | {metrics.calculate_bits_per_base():6.3f} bpb"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines), encoding="utf-8")

    def generate_csv_report(self, results: Sequence[CompressionMetrics], output_path: Path) -> None:
        save_results_csv([metrics.to_dict() for metrics in results], output_path)

    def generate_memory_report(self, results: Sequence[CompressionMetrics], output_path: Path) -> None:
        grouped: Dict[str, List[CompressionMetrics]] = {}
        for metrics in results:
            grouped.setdefault(metrics.dataset, []).append(metrics)
        lines = ["MEMORY USAGE ANALYSIS", "====================", ""]
        for dataset, items in grouped.items():
            lines.append(f"Dataset: {dataset}")
            for metrics in items:
                lines.append(f"{metrics.algorithm}:")
                lines.append(f"  Peak Memory Usage:   {metrics.peak_memory:.2f} MB")
                lines.append(f"  Dictionary Size:     {metrics.dictionary_size:.2f} KB")
                lines.append(f"  Memory per Base:     {metrics.peak_memory / metrics.original_bases:.6f} MB")
            lines.append("")
        lines.extend(
            [
                "Analysis:",
                "  Lower peak memory and smaller dictionary overhead are preferable for large datasets.",
                "  VCSD+ uses a trie dictionary; LZSS uses a fixed sliding window.",
            ]
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines), encoding="utf-8")

    def generate_json_report(self, results: Sequence[CompressionMetrics], output_path: Path) -> None:
        payload = {
            "benchmark_metadata": {
                "timestamp": current_timestamp(),
                "total_runtime_seconds": sum(metric.encode_time + metric.decode_time for metric in results),
            },
            "algorithms": {},
        }
        for metrics in results:
            payload["algorithms"].setdefault(metrics.algorithm, {})[metrics.dataset] = {
                "encode_time_s": metrics.encode_time,
                "decode_time_s": metrics.decode_time,
                "total_time_s": metrics.encode_time + metrics.decode_time,
                "compression_ratio_%": metrics.calculate_compression_ratio(),
                "bits_per_base": metrics.calculate_bits_per_base(),
                "peak_memory_mb": metrics.peak_memory,
                "lossless": metrics.lossless,
            }
        save_results_json(payload, output_path)

    def generate_comparative_summary(self, results: Sequence[CompressionMetrics], output_path: Path) -> None:
        grouped: Dict[str, List[CompressionMetrics]] = {}
        for metrics in results:
            grouped.setdefault(metrics.dataset, []).append(metrics)
        lines = [
            "COMPARATIVE ANALYSIS: DNA-specific codecs vs general-purpose baselines",
            "=====================================================================",
            "",
            "DNA-specific: VCSDplus (compact + entropy modes), DNACompress, LZ78, LZSS.",
            "General-purpose: gzip, lzma, bz2, zlib, 2-bit pack (information-theoretic floor).",
            "",
        ]
        for dataset, items in grouped.items():
            lines.append(f"Dataset: {dataset}")
            lines.append("  All algorithms:")
            for entry in sorted(items, key=lambda item: item.calculate_compression_ratio()):
                lines.append(
                    f"    - {entry.algorithm}: "
                    f"ratio={entry.calculate_compression_ratio():.2f}% | "
                    f"encode={entry.encode_time:.4f}s | decode={entry.decode_time:.4f}s | "
                    f"lossless={'YES' if entry.lossless else 'NO'}"
                )
            best = min(items, key=lambda item: item.calculate_compression_ratio())
            lines.append(f"  Best compression: {best.algorithm} ({best.calculate_compression_ratio():.2f}%)")
            fastest = min(items, key=lambda item: item.encode_time + item.decode_time)
            lines.append(f"  Fastest overall:   {fastest.algorithm} ({fastest.encode_time + fastest.decode_time:.4f}s)")
            lines.append("")
        lines.extend(
            [
                "READING GUIDE",
                "  - 2-bit pack at ~25% ratio is the information-theoretic floor for i.i.d.",
                "    DNA; any 'compressor' that does not beat it on a given dataset is",
                "    not actually compressing relative to the trivial encoding.",
                "  - The general-purpose baselines (lzma / bz2 / zlib / gzip) consistently",
                "    beat the DNA-specific codecs on synthetic repetitive data.  The",
                "    DNA-specific codecs win only on inputs where their structural",
                "    assumption (long exact repeats for DNACompress, codon-scale repetition",
                "    for VCSD+) actually holds; see Results/ablation_results.txt for",
                "    a per-flag decomposition.",
                "  - Encode and decode times for VCSDplus on lambda_phage are O(seconds)",
                "    in pure Python; production-scale comparisons would need a port to a",
                "    faster language.",
            ]
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines), encoding="utf-8")

    def generate_entropy_report(
        self,
        datasets: Sequence[Tuple[str, Sequence[SequenceRecord]]],
        output_path: Path,
    ) -> None:
        """Per-dataset GC content and order-0..5 empirical entropy.

        Use this to read every ``bits_per_base`` value in the other reports
        in context: H_k is the lower bound on what any order-k predictor can
        achieve on this exact input.
        """
        lines = ["INFORMATION-THEORETIC ANALYSIS", "==============================", ""]
        for name, records in datasets:
            text = "".join(seq for _, seq in records)
            lines.append(entropy_report(text, label=name))
            lines.append("")
        lines.extend([
            "Reading guide:",
            "  - For i.i.d. uniform DNA the floor is exactly 2.0 bits/symbol (H_0).",
            "  - H_k is non-increasing in k; the gap between H_k and the codec's",
            "    achieved bpb measures how much higher-order structure that codec",
            "    is leaving on the table.",
            "  - 2-bit packing achieves H_0 exactly (with negligible padding).",
        ])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines), encoding="utf-8")


def prepare_data() -> List[Path]:
    SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    created: List[Path] = []
    for filename, config in DEFAULT_DATASETS.items():
        path = SAMPLE_DATA_DIR / filename
        records = generate_dataset(
            num_sequences=config["num_sequences"],
            avg_length=config["avg_length"],
            dataset_type=config["dataset_type"],
            seed=42,
        )
        save_sequences_to_fasta(records, path)
        created.append(path)
    return created


def load_datasets() -> List[Tuple[str, List[SequenceRecord]]]:
    prepare_data()
    datasets: List[Tuple[str, List[SequenceRecord]]] = []
    # Synthetic datasets first, then real genomes if present.
    synthetic = sorted(SAMPLE_DATA_DIR.glob("*.fasta"))
    for path in synthetic:
        if path.name in REAL_GENOME_FILES:
            continue
        datasets.append((path.name, load_sequences_from_fasta(path)))
    for filename in REAL_GENOME_FILES:
        path = SAMPLE_DATA_DIR / filename
        if path.exists():
            datasets.append((path.name, load_sequences_from_fasta(path)))
    return datasets


def run_benchmark() -> List[CompressionMetrics]:
    benchmark = CompressionBenchmark()
    results: List[CompressionMetrics] = []
    for dataset_name, dataset in load_datasets():
        results.extend(benchmark.run_all_algorithms(dataset, dataset_name))
    return results


def generate_reports(
    results: Sequence[CompressionMetrics],
    datasets: Optional[Sequence[Tuple[str, Sequence[SequenceRecord]]]] = None,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    reporter = ResultReporter()
    reporter.generate_text_report(results, RESULTS_DIR / "compression_ratio_report.txt")
    reporter.generate_csv_report(results, RESULTS_DIR / "performance_metrics.csv")
    reporter.generate_json_report(results, RESULTS_DIR / "timing_results.json")
    reporter.generate_memory_report(results, RESULTS_DIR / "memory_usage_analysis.txt")
    reporter.generate_comparative_summary(results, RESULTS_DIR / "comparative_analysis.txt")
    if datasets is not None:
        reporter.generate_entropy_report(datasets, RESULTS_DIR / "entropy_analysis.txt")


def full_pipeline() -> List[CompressionMetrics]:
    prepare_data()
    datasets = load_datasets()
    benchmark = CompressionBenchmark()
    results: List[CompressionMetrics] = []
    for dataset_name, dataset in datasets:
        results.extend(benchmark.run_all_algorithms(dataset, dataset_name))
    generate_reports(results, datasets=datasets)
    try:
        run_plots()
    except Exception as exc:  # noqa: BLE001 — plots are non-essential
        print(f"WARNING: plot generation failed: {exc}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="VCSD+ comparison and benchmarking pipeline")
    parser.add_argument("--prepare-data", action="store_true", help="Generate sample FASTA datasets")
    parser.add_argument("--run-benchmark", action="store_true", help="Run all algorithms on the datasets")
    parser.add_argument("--generate-reports", action="store_true", help="Generate benchmark reports from the latest run")
    parser.add_argument("--full-pipeline", action="store_true", help="Run data preparation, benchmarking, and report generation")
    args = parser.parse_args()

    if args.prepare_data:
        prepare_data()
    if args.run_benchmark:
        results = run_benchmark()
        print(json.dumps([metrics.to_dict() for metrics in results], indent=2))
    if args.generate_reports:
        datasets = load_datasets()
        benchmark = CompressionBenchmark()
        results = []
        for dataset_name, dataset in datasets:
            results.extend(benchmark.run_all_algorithms(dataset, dataset_name))
        generate_reports(results, datasets=datasets)
    if args.full_pipeline or not any(vars(args).values()):
        results = full_pipeline()
        print(json.dumps([metrics.to_dict() for metrics in results], indent=2))


if __name__ == "__main__":
    main()
