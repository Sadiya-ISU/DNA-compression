"""Ablation study for VCSD+ feature flags.

Runs every combination of the four feature flags
(``sequence_ordering``, ``use_approximate``, ``use_reverse_complement``,
``use_token_entropy``) on each dataset and reports compressed size and
encode/decode time.  The flag with the largest size delta on each dataset
identifies the single most-impactful component for that input class.

Output: ``Results/ablation_results.txt`` (human-readable table) and
``Results/ablation_results.csv`` (machine-readable, one row per
configuration × dataset).
"""
from __future__ import annotations

import csv
import itertools
import time
from pathlib import Path
from typing import List, Sequence, Tuple

from VCSDplus import VCSDDecoder, VCSDEncoder
from utils import SequenceRecord, load_sequences_from_fasta

PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"
RESULTS_DIR = PROJECT_ROOT / "Results"

FLAGS = (
    "sequence_ordering",
    "use_approximate",
    "use_reverse_complement",
    "use_token_entropy",
)


def _config_name(flags: dict) -> str:
    enabled = [k for k, v in flags.items() if v]
    return "+".join(enabled) if enabled else "compact_baseline"


def _run_single(records: Sequence[SequenceRecord], flags: dict) -> Tuple[int, float, float, bool]:
    encoder = VCSDEncoder(**flags)
    decoder = VCSDDecoder()
    t0 = time.perf_counter()
    encoded = encoder.encode(records)
    encode_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    decoded = decoder.decode(encoded)
    decode_time = time.perf_counter() - t0
    decoded_text = "".join(seq for _, seq in decoded)
    expected_text = "".join(seq for _, seq in records)
    return len(encoded), encode_time, decode_time, decoded_text == expected_text


def all_flag_combinations() -> List[dict]:
    return [
        dict(zip(FLAGS, combo))
        for combo in itertools.product([False, True], repeat=len(FLAGS))
    ]


def run_ablation(dataset_paths: Sequence[Path]) -> List[dict]:
    rows: List[dict] = []
    for path in dataset_paths:
        if not path.exists():
            continue
        records = load_sequences_from_fasta(path)
        original_bytes = sum(len(seq) for _, seq in records)
        if original_bytes == 0:
            continue
        for flags in all_flag_combinations():
            try:
                size, et, dt, lossless = _run_single(records, flags)
            except Exception as exc:  # noqa: BLE001 — log and continue
                rows.append({
                    "dataset": path.name,
                    "config": _config_name(flags),
                    "compressed_bytes": -1,
                    "compression_ratio_pct": -1.0,
                    "bits_per_base": -1.0,
                    "encode_time_s": -1.0,
                    "decode_time_s": -1.0,
                    "lossless": False,
                    "error": str(exc),
                    **{flag: flags[flag] for flag in FLAGS},
                })
                continue
            rows.append({
                "dataset": path.name,
                "config": _config_name(flags),
                "compressed_bytes": size,
                "compression_ratio_pct": size / original_bytes * 100.0,
                "bits_per_base": size * 8.0 / original_bytes,
                "encode_time_s": et,
                "decode_time_s": dt,
                "lossless": lossless,
                "error": "",
                **{flag: flags[flag] for flag in FLAGS},
            })
    return rows


def write_text_report(rows: Sequence[dict], path: Path) -> None:
    by_dataset: dict[str, list[dict]] = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], []).append(row)
    lines = ["VCSD+ ABLATION STUDY", "====================", ""]
    for dataset, dataset_rows in by_dataset.items():
        baseline = next(r for r in dataset_rows if r["config"] == "compact_baseline")
        baseline_size = baseline["compressed_bytes"]
        lines.append(f"Dataset: {dataset}")
        lines.append(f"  Baseline (no flags): {baseline_size} B  ratio={baseline['compression_ratio_pct']:.2f}%  bpb={baseline['bits_per_base']:.3f}")
        lines.append("  All configurations (sorted by compressed size):")
        for row in sorted(dataset_rows, key=lambda r: r["compressed_bytes"]):
            delta_bytes = row["compressed_bytes"] - baseline_size
            delta_pct = (delta_bytes / baseline_size * 100.0) if baseline_size else 0.0
            lines.append(
                f"    {row['config']:<60s} "
                f"{row['compressed_bytes']:>8d} B   "
                f"ratio={row['compression_ratio_pct']:6.2f}%   "
                f"bpb={row['bits_per_base']:5.3f}   "
                f"{delta_pct:+6.2f}%   "
                f"lossless={'YES' if row['lossless'] else 'NO'}"
            )
        # Per-flag marginal: average size delta when toggling each flag with all
        # others held at their average (the standard one-factor-at-a-time read).
        flag_deltas: dict[str, list[float]] = {flag: [] for flag in FLAGS}
        for off_row in dataset_rows:
            for flag in FLAGS:
                if not off_row[flag]:
                    on_flags = {**off_row, flag: True}
                    on_match = next(
                        (r for r in dataset_rows if all(r[f] == on_flags[f] for f in FLAGS)),
                        None,
                    )
                    if on_match and off_row["compressed_bytes"] > 0:
                        delta = (
                            (on_match["compressed_bytes"] - off_row["compressed_bytes"])
                            / off_row["compressed_bytes"]
                            * 100.0
                        )
                        flag_deltas[flag].append(delta)
        lines.append("  Average marginal contribution per flag (negative = smaller output):")
        for flag, deltas in flag_deltas.items():
            if deltas:
                mean = sum(deltas) / len(deltas)
                lines.append(f"    {flag:<26s}: {mean:+.2f}%  (averaged over {len(deltas)} pairs)")
        lines.append("")
    lines.extend([
        "Reading guide:",
        "  Each row reports compressed_bytes for one (dataset, flag-combo) pair.",
        "  The 'marginal contribution' lines isolate each flag by averaging the",
        "  size change from toggling it on while holding the other three fixed.",
        "  A flag whose marginal is consistently near 0% across datasets is",
        "  not earning its keep on this benchmark.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv_report(rows: Sequence[dict], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    paths = sorted(SAMPLE_DATA_DIR.glob("*.fasta"))
    if not paths:
        print("No FASTA datasets found; run comparison.py --prepare-data first.")
        return
    print(f"Running ablation on {len(paths)} datasets x {2**len(FLAGS)} configs...")
    rows = run_ablation(paths)
    write_text_report(rows, RESULTS_DIR / "ablation_results.txt")
    write_csv_report(rows, RESULTS_DIR / "ablation_results.csv")
    print(f"Wrote {RESULTS_DIR / 'ablation_results.txt'} and ablation_results.csv")
    bad = [r for r in rows if not r["lossless"]]
    if bad:
        print(f"WARNING: {len(bad)} non-lossless configs detected.")
        for row in bad[:5]:
            print(f"  - {row['dataset']} / {row['config']}")
    else:
        print("All configs lossless.")


if __name__ == "__main__":
    main()
