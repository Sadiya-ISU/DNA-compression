"""LZ78-style dictionary compression for DNA sequences."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Sequence, Tuple, Union

from utils import SequenceRecord, combine_sequences

MAGIC = b"LZ78D1"
BASE_TO_BYTE = {base: index for index, base in enumerate("ACGT")}
BYTE_TO_BASE = {index: base for base, index in BASE_TO_BYTE.items()}


@dataclass
class LZ78Token:
    """LZ78 token containing a phrase reference and an appended base."""

    phrase_id: int
    appended_base: str


@dataclass
class _TrieNode:
    children: Dict[str, "_TrieNode"]
    phrase_id: int = 0

    def __init__(self) -> None:
        self.children = {}
        self.phrase_id = 0


class LZ78Dictionary:
    """Dynamic dictionary used by the LZ78-style encoder."""

    def __init__(self, max_size: int | None = None) -> None:
        self.max_size = max_size
        self.phrases: List[str] = [""]
        self.phrase_to_id: Dict[str, int] = {"": 0}
        self.root = _TrieNode()

    def add_phrase(self, phrase: str) -> int:
        if not phrase:
            return 0
        if self.max_size is not None and len(self.phrases) >= self.max_size:
            return self.phrase_to_id.get(phrase, 0)
        if phrase not in self.phrase_to_id:
            self.phrases.append(phrase)
            self.phrase_to_id[phrase] = len(self.phrases) - 1
            node = self.root
            for char in phrase:
                node = node.children.setdefault(char, _TrieNode())
            node.phrase_id = self.phrase_to_id[phrase]
        return self.phrase_to_id[phrase]

    def find_longest_match(self, data: str, position: int) -> Tuple[int, str]:
        node = self.root
        best_id = 0
        best_phrase = ""
        index = position
        while index < len(data):
            char = data[index]
            if char not in node.children:
                break
            node = node.children[char]
            if node.phrase_id:
                best_id = node.phrase_id
                best_phrase = data[position : index + 1]
            index += 1
        return best_id, best_phrase

    def get_phrase_by_id(self, phrase_id: int) -> str:
        if phrase_id < 0 or phrase_id >= len(self.phrases):
            raise KeyError(phrase_id)
        return self.phrases[phrase_id]


class LZ78Encoder:
    """Greedy LZ78-style encoder."""

    def __init__(self, dictionary_max_size: int | None = None) -> None:
        self.dictionary = LZ78Dictionary(dictionary_max_size)
        self.last_tokens: List[LZ78Token] = []

    def emit_token(self, phrase_id: int, next_char: str) -> LZ78Token:
        return LZ78Token(phrase_id, next_char)

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
        if not text:
            return MAGIC + b"\x00"

        tokens: List[LZ78Token] = []
        position = 0
        while position < len(text):
            phrase_id, phrase = self.dictionary.find_longest_match(text, position)
            next_index = position + len(phrase)
            if next_index < len(text):
                appended_base = text[next_index]
                token = self.emit_token(phrase_id, appended_base)
                tokens.append(token)
                self.dictionary.add_phrase(phrase + appended_base)
                position = next_index + 1
            else:
                token = self.emit_token(phrase_id, "")
                tokens.append(token)
                if phrase:
                    self.dictionary.add_phrase(phrase)
                position = len(text)

        self.last_tokens = tokens
        buffer = BytesIO()
        buffer.write(MAGIC)
        self._write_varint(buffer, len(text))
        self._write_varint(buffer, len(tokens))
        for token in tokens:
            self._write_varint(buffer, token.phrase_id)
            if token.appended_base:
                buffer.write(bytes([1]))
                buffer.write(bytes([BASE_TO_BYTE[token.appended_base]]))
            else:
                buffer.write(bytes([0]))
        return buffer.getvalue()


class LZ78Decoder:
    """Decoder for LZ78-style token streams."""

    def __init__(self) -> None:
        self.dictionary = LZ78Dictionary()

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
            raise ValueError("Invalid LZ78 stream")
        original_length = self._read_varint(buffer)
        token_count = self._read_varint(buffer)
        output: List[str] = []
        for _ in range(token_count):
            phrase_id = self._read_varint(buffer)
            flag = buffer.read(1)
            if not flag:
                raise EOFError("Unexpected end of stream")
            appended_base = ""
            if flag[0]:
                byte = buffer.read(1)
                if not byte:
                    raise EOFError("Unexpected end of stream")
                appended_base = BYTE_TO_BASE[byte[0]]
            phrase = self.dictionary.get_phrase_by_id(phrase_id)
            reconstructed = phrase + appended_base
            if reconstructed:
                output.append(reconstructed)
                self.dictionary.add_phrase(reconstructed)
        text = "".join(output)
        return text[:original_length]


__all__ = ["LZ78Token", "LZ78Dictionary", "LZ78Encoder", "LZ78Decoder"]
