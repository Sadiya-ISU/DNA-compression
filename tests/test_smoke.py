"""Smoke test confirming the test harness loads the project correctly."""

def test_import_vcsd():
    from VCSDplus import VCSDEncoder, VCSDDecoder  # noqa: F401


def test_fixture_random_dna(random_dna_short):
    assert len(random_dna_short) == 120
    assert set(random_dna_short).issubset(set("ACGT"))
