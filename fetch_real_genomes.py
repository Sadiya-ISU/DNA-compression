"""Download small public reference genomes for the benchmark.

The two genomes (Lambda phage NC_001416, ϕX174 NC_001422) are public
domain, small (5 KB and 48 KB respectively), and well-suited to pure-Python
codec benchmarks.  Files are cached under sample_data/ so the network is
contacted at most once.

Run:
    python3 fetch_real_genomes.py

Skips the download if both files already exist.  If the network is
unavailable the rest of the project still runs on synthetic data.
"""
from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"

# NCBI E-utils efetch endpoint, which serves a FASTA-formatted record per ID.
NCBI_EFETCH = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    "?db=nuccore&id={accession}&rettype=fasta&retmode=text"
)

GENOMES = [
    # The fourth column is a generous size cap (~2x the expected return)
    # to defend against unexpectedly large records without false positives.
    ("lambda_phage.fasta", "NC_001416.1", 48_502, 100_000),
    ("phiX174.fasta",       "NC_001422.1",  5_386,  20_000),
]


def _download(accession: str) -> str:
    url = NCBI_EFETCH.format(accession=accession)
    request = urllib.request.Request(url, headers={"User-Agent": "compression-benchmark/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("ascii")


def _fasta_total_bases(text: str) -> int:
    total = 0
    for line in text.splitlines():
        if not line or line.startswith(">"):
            continue
        total += sum(1 for c in line.strip() if c in "ACGTacgt")
    return total


def fetch_all() -> int:
    SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    failures = 0
    for filename, accession, expected_bp, max_bytes in GENOMES:
        target = SAMPLE_DATA_DIR / filename
        # Treat anything below half the expected length as a corrupt stub
        # and re-fetch.  100-byte heuristic was too lax: a 200-byte partial
        # download would have been wrongly accepted as cached.
        min_cached_bytes = max(100, expected_bp // 2)
        if target.exists() and target.stat().st_size > min_cached_bytes:
            print(f"[skip] {target} (already cached, {target.stat().st_size} B)")
            continue
        print(f"[fetch] {accession} -> {target}")
        try:
            text = _download(accession)
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"[FAIL] {accession}: {exc}", file=sys.stderr)
            failures += 1
            continue
        if len(text.encode("ascii")) > max_bytes:
            print(f"[FAIL] {accession} returned >{max_bytes} B; skipping", file=sys.stderr)
            failures += 1
            continue
        total = _fasta_total_bases(text)
        # Soft sanity: NCBI sometimes returns a slightly different lineage strain
        # whose length differs from the reference; warn but accept within 5%.
        if abs(total - expected_bp) > expected_bp * 0.05:
            print(
                f"[warn] {accession} length {total} differs by >5% from expected {expected_bp}",
                file=sys.stderr,
            )
        target.write_text(text, encoding="ascii")
        print(f"[done] {target} ({total} bp)")
    return failures


if __name__ == "__main__":
    sys.exit(fetch_all())
