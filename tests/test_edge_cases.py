"""Edge-case tests for all four codecs."""
from __future__ import annotations

import pytest

from DNACompress import DNACompressDecoder, DNACompressEncoder
from LZ78_style import LZ78Decoder, LZ78Encoder
from LZSS import LZSSDecoder, LZSSEncoder
from VCSDplus import VCSDDecoder, VCSDEncoder

# ---- single-base / very short inputs ----------------------------------------


@pytest.mark.parametrize("text", ["A", "C", "G", "T", "AC", "ACG", "ACGT"])
def test_dnacompress_short_inputs(text):
    encoder, decoder = DNACompressEncoder(), DNACompressDecoder()
    assert decoder.decode(encoder.encode(text)) == text


@pytest.mark.parametrize("text", ["A", "C", "G", "T", "AC", "ACG", "ACGT"])
def test_lz78_short_inputs(text):
    encoder, decoder = LZ78Encoder(), LZ78Decoder()
    assert decoder.decode(encoder.encode(text)) == text


@pytest.mark.parametrize("text", ["A", "C", "G", "T", "AC", "ACG", "ACGT"])
def test_lzss_short_inputs(text):
    encoder, decoder = LZSSEncoder(), LZSSDecoder()
    assert decoder.decode(encoder.encode(text)) == text


@pytest.mark.parametrize("text", ["ACG", "ACGACG", "ACGACGACG", "AAACCCGGGTTT"])
def test_vcsd_short_inputs(text):
    encoder, decoder = VCSDEncoder(), VCSDDecoder()
    encoded = encoder.encode(text)
    decoded = decoder.decode(encoded)
    assert "".join(seq for _, seq in decoded) == text


# ---- non-codon-aligned inputs (trailing suffix) ------------------------------


@pytest.mark.parametrize("length", [1, 2, 4, 5, 7, 8])
def test_vcsd_codon_unaligned_suffix(length):
    """VCSD+ explicitly handles a trailing 0-2 base suffix; verify all alignments."""
    text = "ACGT" * 4
    text = text[:length]
    encoder, decoder = VCSDEncoder(), VCSDDecoder()
    encoded = encoder.encode(text)
    decoded = decoder.decode(encoded)
    assert "".join(seq for _, seq in decoded) == text


# ---- homopolymer / extreme repetition ----------------------------------------


@pytest.mark.parametrize("base", ["A", "C", "G", "T"])
def test_homopolymer_all_codecs(base):
    text = base * 600
    for encoder_cls, decoder_cls in [
        (DNACompressEncoder, DNACompressDecoder),
        (LZ78Encoder, LZ78Decoder),
        (LZSSEncoder, LZSSDecoder),
    ]:
        assert decoder_cls().decode(encoder_cls().encode(text)) == text
    decoded = VCSDDecoder().decode(VCSDEncoder().encode(text))
    assert "".join(seq for _, seq in decoded) == text


# ---- codec-specific invariants ------------------------------------------------


def test_lzss_match_threshold():
    """Greedy LZSS only emits matches of length >= 3; below that should be literals."""
    text = "ACGT" * 50
    encoder = LZSSEncoder(window_size=64, lookahead_buffer_size=16)
    encoded = encoder.encode(text)
    assert LZSSDecoder().decode(encoded) == text
    # Sanity: at least one match token exists for periodic input.
    has_match = any(token.is_match_flag for token in encoder.last_tokens)
    assert has_match, "LZSS must find at least one match in 'ACGT'*50"


def test_vcsd_invalid_bitstream_rejected():
    """The decoder should reject a stream that doesn't start with the magic bytes."""
    decoder = VCSDDecoder()
    with pytest.raises(ValueError):
        decoder.decode(b"NOTVCSD")


def test_lz78_invalid_bitstream_rejected():
    decoder = LZ78Decoder()
    with pytest.raises(ValueError):
        decoder.decode(b"NOTLZ78")


def test_lzss_invalid_bitstream_rejected():
    decoder = LZSSDecoder()
    with pytest.raises(ValueError):
        decoder.decode(b"NOTLZSS")


def test_dnacompress_invalid_bitstream_rejected():
    decoder = DNACompressDecoder()
    with pytest.raises(ValueError):
        decoder.decode(b"NOTDNAC")
