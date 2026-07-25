"""LZSS sliding-window compression for DNA sequences."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import List, Tuple, Union

from utils import SequenceRecord, combine_sequences

MAGIC = b"LZSSD1"
BASE_TO_BYTE = {base: index for index, base in enumerate("ACGT")}
BYTE_TO_BASE = {index: base for base, index in BASE_TO_BYTE.items()}


@dataclass
class LZSSToken:
    """Token representation for LZSS."""

    is_match_flag: bool
    offset: int = 0
    length: int = 0
    literal: str = ""


class SlidingWindow:
    """Fixed-size sliding window for match search."""

    def __init__(self, window_size: int = 16384, lookahead_buffer_size: int = 258) -> None:
        self.window_size = window_size
        self.lookahead_buffer_size = lookahead_buffer_size

    def search(self, data: str, position: int) -> Tuple[int, int]:
        window_start = max(0, position - self.window_size)
        best_offset = 0
        best_length = 0
        max_length = min(self.lookahead_buffer_size, len(data) - position)
        for candidate_start in range(window_start, position):
            length = 0
            while (
                length < max_length
                and candidate_start + length < position
                and position + length < len(data)
                and data[candidate_start + length] == data[position + length]
            ):
                length += 1
            if length > best_length:
                best_length = length
                best_offset = position - candidate_start
        return best_offset, best_length


class LZSSEncoder:
    """Greedy LZSS encoder."""

    def __init__(self, window_size: int = 16384, lookahead_buffer_size: int = 258) -> None:
        self.window = SlidingWindow(window_size, lookahead_buffer_size)
        self.last_tokens: List[LZSSToken] = []

    def emit_match_token(self, offset: int, length: int) -> LZSSToken:
        return LZSSToken(True, offset=offset, length=length)

    def emit_literal_token(self, char: str) -> LZSSToken:
        return LZSSToken(False, literal=char)

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
        tokens: List[LZSSToken] = []
        position = 0
        while position < len(text):
            offset, length = self.window.search(text, position)
            if length >= 3:
                tokens.append(self.emit_match_token(offset, length))
                position += length
            else:
                tokens.append(self.emit_literal_token(text[position]))
                position += 1
        self.last_tokens = tokens

        buffer = BytesIO()
        buffer.write(MAGIC)
        self._write_varint(buffer, len(text))
        self._write_varint(buffer, len(tokens))
        for token in tokens:
            if token.is_match_flag:
                buffer.write(bytes([1]))
                self._write_varint(buffer, token.offset)
                self._write_varint(buffer, token.length)
            else:
                buffer.write(bytes([0]))
                buffer.write(bytes([BASE_TO_BYTE[token.literal]]))
        return buffer.getvalue()


class LZSSDecoder:
    """Decoder for LZSS token streams."""

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
            raise ValueError("Invalid LZSS stream")
        original_length = self._read_varint(buffer)
        token_count = self._read_varint(buffer)
        output: List[str] = []
        for _ in range(token_count):
            flag = buffer.read(1)
            if not flag:
                raise EOFError("Unexpected end of stream")
            if flag[0]:
                offset = self._read_varint(buffer)
                length = self._read_varint(buffer)
                if offset <= 0 or length <= 0 or offset > len(output):
                    raise ValueError("Invalid match token")
                start = len(output) - offset
                for index in range(length):
                    output.append(output[start + index])
            else:
                byte = buffer.read(1)
                if not byte:
                    raise EOFError("Unexpected end of stream")
                output.append(BYTE_TO_BASE[byte[0]])
        return "".join(output)[:original_length]


__all__ = ["LZSSToken", "SlidingWindow", "LZSSEncoder", "LZSSDecoder"]
