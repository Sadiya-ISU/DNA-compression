"""Standard general-purpose compressor baselines.

These are *not* novel — they are stdlib wrappers around gzip/lzma/bz2/zlib —
but they are essential controls: any DNA-specific codec that does not beat
gzip on real genomic data has not earned its claim of being DNA-specific.

The classes intentionally mirror the (encode, decode) interface used by the
LZ78, LZSS, and DNACompress codecs in this project so they can be plugged
into ``CompressionBenchmark`` without special-case logic.
"""
from __future__ import annotations

import bz2
import gzip
import lzma
import zlib
from typing import Sequence, Union

from utils import SequenceRecord, combine_sequences


def _input_to_text(data: Union[str, Sequence[SequenceRecord]]) -> str:
    if isinstance(data, str):
        return data.strip().upper()
    return combine_sequences(sequence for _, sequence in data)


class GzipEncoder:
    """gzip (DEFLATE) wrapper at maximum compression level."""

    def __init__(self, level: int = 9) -> None:
        self.level = level

    def encode(self, data: Union[str, Sequence[SequenceRecord]]) -> bytes:
        text = _input_to_text(data)
        return gzip.compress(text.encode("ascii"), compresslevel=self.level)


class GzipDecoder:
    def decode(self, encoded: bytes) -> str:
        return gzip.decompress(encoded).decode("ascii")


class LzmaEncoder:
    """LZMA / XZ wrapper at default preset."""

    def __init__(self, preset: int = lzma.PRESET_DEFAULT) -> None:
        self.preset = preset

    def encode(self, data: Union[str, Sequence[SequenceRecord]]) -> bytes:
        text = _input_to_text(data)
        return lzma.compress(text.encode("ascii"), preset=self.preset)


class LzmaDecoder:
    def decode(self, encoded: bytes) -> str:
        return lzma.decompress(encoded).decode("ascii")


class Bz2Encoder:
    """bzip2 (BWT + RLE + Huffman) wrapper at maximum compression."""

    def __init__(self, compresslevel: int = 9) -> None:
        self.compresslevel = compresslevel

    def encode(self, data: Union[str, Sequence[SequenceRecord]]) -> bytes:
        text = _input_to_text(data)
        return bz2.compress(text.encode("ascii"), compresslevel=self.compresslevel)


class Bz2Decoder:
    def decode(self, encoded: bytes) -> str:
        return bz2.decompress(encoded).decode("ascii")


class ZlibEncoder:
    """Raw zlib (DEFLATE without gzip framing) wrapper at maximum compression.

    Included separately from gzip so the report can quantify how much of the
    'VCSDplus_entropy' improvement comes from the zlib pass it bolts on at
    the end.
    """

    def __init__(self, level: int = 9) -> None:
        self.level = level

    def encode(self, data: Union[str, Sequence[SequenceRecord]]) -> bytes:
        text = _input_to_text(data)
        return zlib.compress(text.encode("ascii"), level=self.level)


class ZlibDecoder:
    def decode(self, encoded: bytes) -> str:
        return zlib.decompress(encoded).decode("ascii")


class TwoBitEncoder:
    """Trivial 2-bit packer of A/C/G/T. The information-theoretic floor for
    i.i.d. uniform DNA. Useful as a per-base lower bound on any unstructured
    encoding and a sanity check on every codec's bits-per-base column.

    Stream layout (designed for round-trip, not for production use):
        - 4 byte big-endian length prefix (number of bases)
        - ceil(length/4) packed bytes, 2 bits per base, MSB first
    """

    BASE_BITS = {"A": 0b00, "C": 0b01, "G": 0b10, "T": 0b11}

    def encode(self, data: Union[str, Sequence[SequenceRecord]]) -> bytes:
        text = _input_to_text(data)
        out = bytearray(len(text).to_bytes(4, "big"))
        bits = 0
        bit_count = 0
        for base in text:
            bits = (bits << 2) | self.BASE_BITS[base]
            bit_count += 2
            if bit_count == 8:
                out.append(bits)
                bits = 0
                bit_count = 0
        if bit_count:
            out.append(bits << (8 - bit_count))
        return bytes(out)


class TwoBitDecoder:
    BITS_BASE = {0b00: "A", 0b01: "C", 0b10: "G", 0b11: "T"}

    def decode(self, encoded: bytes) -> str:
        if len(encoded) < 4:
            raise ValueError("TwoBit stream too short")
        length = int.from_bytes(encoded[:4], "big")
        body = encoded[4:]
        bases = []
        for byte in body:
            for shift in (6, 4, 2, 0):
                bases.append(self.BITS_BASE[(byte >> shift) & 0b11])
                if len(bases) == length:
                    return "".join(bases)
        return "".join(bases[:length])


__all__ = [
    "GzipEncoder", "GzipDecoder",
    "LzmaEncoder", "LzmaDecoder",
    "Bz2Encoder", "Bz2Decoder",
    "ZlibEncoder", "ZlibDecoder",
    "TwoBitEncoder", "TwoBitDecoder",
]
