"""DNACompress-like repeat-aware compressor for DNA sequences.

This module implements a closer DNACompress concept than the initial baseline:
- exact repeat tokens
- reverse-complement repeat tokens
- bounded approximate repeat tokens with mismatch patches

It remains a practical reference implementation, not a byte-for-byte replica
of any historical binary format.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from typing import Dict, List, Sequence, Tuple, Union

from utils import SequenceRecord, combine_sequences, reverse_complement

MAGIC = b"DNACMP2"
BASE_TO_CODE = {"A": 0, "C": 1, "G": 2, "T": 3}
CODE_TO_BASE = {value: key for key, value in BASE_TO_CODE.items()}
COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}

TOKEN_LITERAL = 0
TOKEN_EXACT = 1
TOKEN_APPROX = 2
TOKEN_RC_EXACT = 3
TOKEN_RC_APPROX = 4


@dataclass
class DNACompressToken:
    """Token representation for DNACompress-like output."""

    token_type: int
    offset: int = 0
    length: int = 0
    seed_length: int = 0
    literal: str = ""
    mismatches: List[Tuple[int, str]] = field(default_factory=list)


class RepeatIndex:
    """Seed index for forward and reverse-complement lookup."""

    def __init__(self, seed_length: int = 8, max_candidates: int = 64) -> None:
        self.seed_length = seed_length
        self.max_candidates = max_candidates
        self.forward_positions: Dict[str, List[int]] = {}
        self.rc_positions: Dict[str, List[int]] = {}

    def add_text(self, text: str, start: int, end: int) -> None:
        # Caller has just consumed text[start:end]. Index every seed whose
        # right edge falls in (start, end] — i.e., seeds at positions p with
        # start - seed_length < p <= end - seed_length. This guarantees the
        # full seed text[p : p+seed_length] is within the consumed prefix.
        first = max(0, start - self.seed_length + 1)
        last = end - self.seed_length
        for position in range(first, last + 1):
            if position < 0 or position + self.seed_length > len(text):
                continue
            seed = text[position : position + self.seed_length]
            bucket = self.forward_positions.setdefault(seed, [])
            bucket.append(position)
            if len(bucket) > self.max_candidates:
                del bucket[0]
            rc_bucket = self.rc_positions.setdefault(reverse_complement(seed), [])
            rc_bucket.append(position)
            if len(rc_bucket) > self.max_candidates:
                del rc_bucket[0]

    def forward_candidates(self, text: str, position: int) -> List[int]:
        if position + self.seed_length > len(text):
            return []
        seed = text[position : position + self.seed_length]
        return self.forward_positions.get(seed, [])

    def rc_candidates(self, text: str, position: int) -> List[int]:
        if position + self.seed_length > len(text):
            return []
        seed = text[position : position + self.seed_length]
        return self.rc_positions.get(seed, [])

    def best_forward_match(
        self,
        text: str,
        position: int,
        max_mismatches: int,
    ) -> DNACompressToken:
        best = DNACompressToken(token_type=TOKEN_LITERAL, literal=text[position])
        for candidate_start in reversed(self.forward_candidates(text, position)):
            length = 0
            mismatches = 0
            patches: List[Tuple[int, str]] = []
            while position + length < len(text):
                src_idx = candidate_start + length
                if src_idx >= position:
                    break
                source = text[src_idx]
                target = text[position + length]
                if source != target:
                    mismatches += 1
                    if mismatches > max_mismatches:
                        break
                    patches.append((length, target))
                length += 1
            if length <= best.length:
                continue
            if length < self.seed_length:
                continue
            offset = position - candidate_start
            if mismatches == 0:
                best = DNACompressToken(token_type=TOKEN_EXACT, offset=offset, length=length)
            else:
                best = DNACompressToken(token_type=TOKEN_APPROX, offset=offset, length=length, mismatches=patches)
        return best

    def best_rc_match(
        self,
        text: str,
        position: int,
        max_mismatches: int,
    ) -> DNACompressToken:
        best = DNACompressToken(token_type=TOKEN_LITERAL, literal=text[position])
        for candidate_start in reversed(self.rc_candidates(text, position)):
            length = 0
            mismatches = 0
            patches: List[Tuple[int, str]] = []
            while position + length < len(text):
                src_idx = candidate_start + self.seed_length - 1 - length
                if src_idx < 0:
                    break
                if src_idx >= position:
                    break
                source = COMPLEMENT[text[src_idx]]
                target = text[position + length]
                if source != target:
                    mismatches += 1
                    if mismatches > max_mismatches:
                        break
                    patches.append((length, target))
                length += 1
            if length <= best.length:
                continue
            if length < self.seed_length:
                continue
            source_end = candidate_start + self.seed_length - 1
            offset = (position - 1) - source_end
            if offset < 0:
                continue
            if mismatches == 0:
                best = DNACompressToken(token_type=TOKEN_RC_EXACT, offset=offset, length=length, seed_length=self.seed_length)
            else:
                best = DNACompressToken(
                    token_type=TOKEN_RC_APPROX,
                    offset=offset,
                    length=length,
                    seed_length=self.seed_length,
                    mismatches=patches,
                )
        return best


class DNACompressEncoder:
    """DNACompress-like repeat-aware encoder."""

    def __init__(
        self,
        seed_length: int = 8,
        max_candidates: int = 64,
        min_match_length: int = 8,
        max_mismatches: int = 2,
    ) -> None:
        self.index = RepeatIndex(seed_length=seed_length, max_candidates=max_candidates)
        self.min_match_length = min_match_length
        self.max_mismatches = max_mismatches
        self.last_tokens: List[DNACompressToken] = []

    def _write_varint(self, buffer: BytesIO, value: int) -> None:
        while True:
            byte = value & 0x7F
            value >>= 7
            if value:
                buffer.write(bytes([byte | 0x80]))
            else:
                buffer.write(bytes([byte]))
                break

    def encode(self, data: Union[str, Sequence[SequenceRecord]]) -> bytes:
        if isinstance(data, str):
            text = data.strip().upper()
        else:
            text = combine_sequences(sequence for _, sequence in data)

        tokens: List[DNACompressToken] = []
        position = 0
        while position < len(text):
            forward = self.index.best_forward_match(text, position, self.max_mismatches)
            rc = self.index.best_rc_match(text, position, self.max_mismatches)
            best = forward if forward.length >= rc.length else rc
            if best.length >= self.min_match_length and best.token_type != TOKEN_LITERAL:
                tokens.append(best)
                self.index.add_text(text, position, position + best.length)
                position += best.length
                continue

            literal = DNACompressToken(TOKEN_LITERAL, literal=text[position], length=1)
            tokens.append(literal)
            self.index.add_text(text, position, position + 1)
            position += 1

        self.last_tokens = tokens
        buffer = BytesIO()
        buffer.write(MAGIC)
        self._write_varint(buffer, len(text))
        self._write_varint(buffer, len(tokens))
        for token in tokens:
            buffer.write(bytes([token.token_type]))
            if token.token_type == TOKEN_LITERAL:
                buffer.write(bytes([BASE_TO_CODE[token.literal]]))
            elif token.token_type in {TOKEN_EXACT, TOKEN_RC_EXACT}:
                self._write_varint(buffer, token.offset)
                self._write_varint(buffer, token.length)
            else:
                self._write_varint(buffer, token.offset)
                self._write_varint(buffer, token.length)
                self._write_varint(buffer, len(token.mismatches))
                for mismatch_pos, base in token.mismatches:
                    self._write_varint(buffer, mismatch_pos)
                    buffer.write(bytes([BASE_TO_CODE[base]]))
        return buffer.getvalue()


class DNACompressDecoder:
    """Decoder for DNACompress-like tokens."""

    def _read_varint(self, buffer: BytesIO) -> int:
        value = 0
        shift = 0
        while True:
            byte = buffer.read(1)
            if not byte:
                raise EOFError("Unexpected end of stream")
            byte_value = byte[0]
            value |= (byte_value & 0x7F) << shift
            if not (byte_value & 0x80):
                return value
            shift += 7

    def decode(self, tokens: bytes) -> str:
        buffer = BytesIO(tokens)
        magic = buffer.read(len(MAGIC))
        if magic != MAGIC:
            raise ValueError("Invalid DNACompress stream")
        original_length = self._read_varint(buffer)
        token_count = self._read_varint(buffer)
        output: List[str] = []
        for _ in range(token_count):
            token_type_raw = buffer.read(1)
            if not token_type_raw:
                raise EOFError("Unexpected end of stream")
            token_type = token_type_raw[0]
            if token_type == TOKEN_LITERAL:
                byte = buffer.read(1)
                if not byte:
                    raise EOFError("Unexpected end of stream")
                output.append(CODE_TO_BASE[byte[0]])
            elif token_type in {TOKEN_EXACT, TOKEN_APPROX}:
                offset = self._read_varint(buffer)
                length = self._read_varint(buffer)
                if offset <= 0 or length <= 0 or offset > len(output):
                    raise ValueError("Invalid match token")
                start = len(output) - offset
                chunk = [output[start + index] for index in range(length)]
                if token_type == TOKEN_APPROX:
                    mismatch_count = self._read_varint(buffer)
                    for _ in range(mismatch_count):
                        mismatch_pos = self._read_varint(buffer)
                        base_code = buffer.read(1)
                        if not base_code:
                            raise EOFError("Unexpected end of stream")
                        chunk[mismatch_pos] = CODE_TO_BASE[base_code[0]]
                output.extend(chunk)
            else:
                offset = self._read_varint(buffer)
                length = self._read_varint(buffer)
                source_end = len(output) - 1 - offset
                if source_end < 0 or length <= 0:
                    raise ValueError("Invalid reverse-complement token")
                chunk = [COMPLEMENT[output[source_end - index]] for index in range(length)]
                if token_type == TOKEN_RC_APPROX:
                    mismatch_count = self._read_varint(buffer)
                    for _ in range(mismatch_count):
                        mismatch_pos = self._read_varint(buffer)
                        base_code = buffer.read(1)
                        if not base_code:
                            raise EOFError("Unexpected end of stream")
                        chunk[mismatch_pos] = CODE_TO_BASE[base_code[0]]
                output.extend(chunk)
        return "".join(output)[:original_length]


__all__ = ["DNACompressToken", "DNACompressEncoder", "DNACompressDecoder"]
