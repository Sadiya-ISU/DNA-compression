"""VCSD+ codon-aware compression implementation."""
from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from typing import Dict, List, Optional, Sequence, Tuple, Union
import zlib

from utils import CompressionMetrics, SequenceRecord, codon_chunks, reverse_complement

TOKEN_TYPES = ["EXT", "REF", "AEXT", "AREF", "RCEXT", "RCREF"]
TOKEN_PRIORITY = {name: index for index, name in enumerate(TOKEN_TYPES)}
TOKEN_CODES = {name: index for index, name in enumerate(TOKEN_TYPES)}
TOKEN_NAMES = {index: name for name, index in TOKEN_CODES.items()}

CODON_TO_INDEX: Dict[str, int] = {}
INDEX_TO_CODON: Dict[int, str] = {}
_codon_index = 0
for first in "ACGT":
    for second in "ACGT":
        for third in "ACGT":
            codon = first + second + third
            CODON_TO_INDEX[codon] = _codon_index
            INDEX_TO_CODON[_codon_index] = codon
            _codon_index += 1

MAGIC = b"VCSDP1"
STREAM_COMPACT_EXACT = 0
STREAM_EXTENDED = 1
STREAM_COMPACT_EXACT_ENTROPY = 2


def _zigzag_encode(value: int) -> int:
    return (value << 1) ^ (value >> 31)


def _zigzag_decode(value: int) -> int:
    return (value >> 1) ^ -(value & 1)


@dataclass
class Token:
    """Token representation for VCSD+ encoding."""

    token_type: str
    phrase_id: int = 0
    codon: str = ""
    correction_list: List[str] = field(default_factory=list)


@dataclass
class TrieNode:
    children: Dict[str, "TrieNode"] = field(default_factory=dict)
    phrase_id: int = 0
    phrase: Tuple[str, ...] = field(default_factory=tuple)


class CodonSequence:
    """Represent a DNA sequence as codons and a trailing suffix."""

    def __init__(self, sequence: str) -> None:
        self.sequence = sequence.strip().upper()
        self.codons, self.trailing_suffix = codon_chunks(self.sequence)

    def get_codons(self) -> List[str]:
        return list(self.codons)

    def get_trailing_suffix(self) -> str:
        return self.trailing_suffix

    def reconstruct(self, codons: Optional[Sequence[str]] = None, suffix: Optional[str] = None) -> str:
        codon_text = "".join(codons if codons is not None else self.codons)
        suffix_text = suffix if suffix is not None else self.trailing_suffix
        return codon_text + suffix_text


class Dictionary:
    """Prefix-closed codon dictionary with trie-based matching."""

    def __init__(self) -> None:
        self.root = TrieNode()
        self.phrases: List[Tuple[str, ...]] = []
        self.phrase_to_id: Dict[Tuple[str, ...], int] = {}

    def __len__(self) -> int:
        return len(self.phrases)

    def _insert_phrase(self, phrase: Sequence[str]) -> int:
        node = self.root
        phrase_tuple = tuple(phrase)
        for codon in phrase_tuple:
            node = node.children.setdefault(codon, TrieNode())
            if not node.phrase:
                node.phrase = phrase_tuple[: len(node.phrase) + 1]
        if phrase_tuple not in self.phrase_to_id:
            self.phrases.append(phrase_tuple)
            self.phrase_to_id[phrase_tuple] = len(self.phrases)
        node.phrase_id = self.phrase_to_id[phrase_tuple]
        node.phrase = phrase_tuple
        return node.phrase_id

    def add_phrase(self, phrase: Sequence[str]) -> int:
        if not phrase:
            raise ValueError("phrase must not be empty")
        phrase_tuple = tuple(phrase)
        phrase_id = 0
        for end in range(1, len(phrase_tuple) + 1):
            phrase_id = self._insert_phrase(phrase_tuple[:end])
        return phrase_id

    def get_phrase(self, phrase_id: int) -> Tuple[str, ...]:
        if phrase_id <= 0 or phrase_id > len(self.phrases):
            raise KeyError(f"Unknown phrase_id {phrase_id}")
        return self.phrases[phrase_id - 1]

    def longest_match(self, codon_sequence: Sequence[str], position: int) -> Tuple[int, Tuple[str, ...]]:
        node = self.root
        best_id = 0
        best_phrase: Tuple[str, ...] = tuple()
        index = position
        while index < len(codon_sequence):
            codon = codon_sequence[index]
            if codon not in node.children:
                break
            node = node.children[codon]
            if node.phrase_id:
                best_id = node.phrase_id
                best_phrase = node.phrase
            index += 1
        return best_id, best_phrase

    def approximate_match(self, codon_sequence: Sequence[str], position: int) -> Tuple[int, Tuple[str, ...]]:
        target = list(codon_sequence[position:])
        best_id = 0
        best_phrase: Tuple[str, ...] = tuple()
        for phrase_id, phrase in enumerate(self.phrases, start=1):
            if len(phrase) <= len(best_phrase) or len(phrase) > len(target):
                continue
            mismatch_ok = True
            for phrase_codon, target_codon in zip(phrase, target):
                if sum(1 for left, right in zip(phrase_codon, target_codon) if left != right) > 1:
                    mismatch_ok = False
                    break
            if mismatch_ok:
                best_id = phrase_id
                best_phrase = phrase
        return best_id, best_phrase

    def reverse_complement_match(self, codon_sequence: Sequence[str], position: int) -> Tuple[int, Tuple[str, ...]]:
        target_text = "".join(codon_sequence[position:])
        best_id = 0
        best_phrase: Tuple[str, ...] = tuple()
        for phrase_id, phrase in enumerate(self.phrases, start=1):
            phrase_text = "".join(phrase)
            if len(phrase_text) > len(target_text):
                continue
            if target_text.startswith(reverse_complement(phrase_text)) and len(phrase) > len(best_phrase):
                best_id = phrase_id
                best_phrase = phrase
        return best_id, best_phrase


class VCSDEncoder:
    """Main VCSD+ encoder."""

    def __init__(
        self,
        use_approximate: bool = False,
        use_reverse_complement: bool = False,
        sequence_ordering: bool = False,
        use_token_entropy: bool = False,
    ) -> None:
        self.use_approximate = use_approximate
        self.use_reverse_complement = use_reverse_complement
        self.sequence_ordering = sequence_ordering
        self.use_token_entropy = use_token_entropy
        self.dictionary = Dictionary()
        self.last_tokens: List[List[Token]] = []
        self.last_records: List[SequenceRecord] = []

    def build_candidates(self, codon_sequence: Sequence[str], position: int) -> List[Token]:
        candidates: List[Token] = []
        exact_id, exact_phrase = self.dictionary.longest_match(codon_sequence, position)
        if exact_id:
            extension = codon_sequence[position + len(exact_phrase)] if position + len(exact_phrase) < len(codon_sequence) else ""
            candidates.append(Token("REF", exact_id, extension))
        if self.use_reverse_complement:
            rc_id, rc_phrase = self.dictionary.reverse_complement_match(codon_sequence, position)
            if rc_id:
                extension = codon_sequence[position + len(rc_phrase)] if position + len(rc_phrase) < len(codon_sequence) else ""
                correction_list = list(codon_sequence[position : position + len(rc_phrase)])
                candidates.append(Token("RCREF", rc_id, extension, correction_list))
        if self.use_approximate:
            approx_id, approx_phrase = self.dictionary.approximate_match(codon_sequence, position)
            if approx_id:
                extension = codon_sequence[position + len(approx_phrase)] if position + len(approx_phrase) < len(codon_sequence) else ""
                correction_list = list(codon_sequence[position : position + len(approx_phrase)])
                candidates.append(Token("AREF", approx_id, extension, correction_list))
        if not candidates and position < len(codon_sequence):
            candidates.append(Token("EXT", 0, codon_sequence[position]))
        return candidates

    def select_best_token(self, candidates: Sequence[Token]) -> Token:
        if not candidates:
            raise ValueError("No candidates available")
        return sorted(candidates, key=lambda token: TOKEN_PRIORITY.get(token.token_type, 999))[0]

    def _add_phrase_from_token(self, matched_phrase: Sequence[str], extension: str) -> None:
        phrase = list(matched_phrase)
        if extension:
            phrase.append(extension)
        if phrase:
            self.dictionary.add_phrase(phrase)

    def _encode_sequence(self, sequence: str, original_index: int) -> dict:
        codon_sequence = CodonSequence(sequence)
        codons = codon_sequence.get_codons()
        suffix = codon_sequence.get_trailing_suffix()
        tokens: List[Token] = []
        position = 0
        while position < len(codons):
            token = self.select_best_token(self.build_candidates(codons, position))
            if token.token_type == "EXT":
                run: List[str] = []
                while position < len(codons):
                    match_id, _ = self.dictionary.longest_match(codons, position)
                    if match_id:
                        break
                    run.append(codons[position])
                    position += 1
                token.correction_list = run
                tokens.append(token)
                self.dictionary.add_phrase(run)
                continue

            matched_phrase = list(self.dictionary.get_phrase(token.phrase_id))
            matched_length = len(matched_phrase)
            position += matched_length

            if token.token_type == "REF":
                # REF tokens may absorb following unmatched codons as an
                # extension run, stored in correction_list.
                extension_run: List[str] = []
                while position < len(codons):
                    next_match_id, _ = self.dictionary.longest_match(codons, position)
                    if next_match_id:
                        break
                    extension_run.append(codons[position])
                    position += 1
                token.correction_list = extension_run
                if extension_run:
                    self.dictionary.add_phrase(matched_phrase + extension_run)
            else:
                # AREF / RCREF / AEXT / RCEXT: correction_list (set in
                # build_candidates) already holds the actual emitted codons,
                # which differ from the matched dictionary phrase. Preserve it
                # and register the corrected codons as a new phrase so the
                # encoder and decoder stay in sync.
                actual_codons = list(token.correction_list)
                if actual_codons:
                    self.dictionary.add_phrase(actual_codons)
            tokens.append(token)

        if suffix:
            self.dictionary.add_phrase([suffix])

        return {"original_index": original_index, "sequence_length": len(sequence), "suffix": suffix, "tokens": tokens}

    def encode(self, sequences: Union[str, Sequence[SequenceRecord]]) -> bytes:
        if isinstance(sequences, str):
            records: List[SequenceRecord] = [("sequence_1", sequences)]
        else:
            records = [(name, sequence) for name, sequence in sequences]

        if self.sequence_ordering:
            def _ordering_key(item: Tuple[int, SequenceRecord]) -> Tuple[int, int, str]:
                sequence_text = item[1][1]
                codons, _ = codon_chunks(sequence_text)
                return (len(set(codons)) if codons else 0, -len(sequence_text), item[1][0])

            ordered = sorted(enumerate(records), key=_ordering_key)
        else:
            ordered = list(enumerate(records))

        encoded_records = [self._encode_sequence(sequence, index) for index, (_, sequence) in ordered]
        self.last_records = [(f"sequence_{record['original_index'] + 1}", record["sequence_length"]) for record in encoded_records]
        self.last_tokens = [record["tokens"] for record in encoded_records]
        compact_mode = not self.use_approximate and not self.use_reverse_complement
        return self._serialize(encoded_records, compact_mode, self.use_token_entropy)

    def _write_varint(self, buffer: BytesIO, value: int) -> None:
        while True:
            byte = value & 0x7F
            value >>= 7
            if value:
                buffer.write(bytes([byte | 0x80]))
            else:
                buffer.write(bytes([byte]))
                break

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

    def _write_string(self, buffer: BytesIO, text: str) -> None:
        data = text.encode("ascii")
        self._write_varint(buffer, len(data))
        buffer.write(data)

    def _varint_bytes(self, value: int) -> bytes:
        out = BytesIO()
        self._write_varint(out, value)
        return out.getvalue()

    def _read_string(self, buffer: BytesIO) -> str:
        length = self._read_varint(buffer)
        data = buffer.read(length)
        if len(data) != length:
            raise EOFError("Unexpected end of stream")
        return data.decode("ascii")

    def _serialize_compact_entropy_record(self, buffer: BytesIO, tokens: Sequence[Token]) -> None:
        ext_len_buffer = BytesIO()
        pid_delta_raw = BytesIO()
        ext_codon_raw = bytearray()
        previous_phrase_id = 0

        for token in tokens:
            delta = token.phrase_id - previous_phrase_id
            previous_phrase_id = token.phrase_id
            pid_delta_raw.write(self._varint_bytes(_zigzag_encode(delta)))

            self._write_varint(ext_len_buffer, len(token.correction_list))
            for extension_codon in token.correction_list:
                ext_codon_raw.append(CODON_TO_INDEX[extension_codon])

        ext_len_bytes = ext_len_buffer.getvalue()
        pid_delta_bytes = pid_delta_raw.getvalue()
        ext_codon_bytes = bytes(ext_codon_raw)

        compressed_pid_deltas = zlib.compress(pid_delta_bytes, level=1)
        compressed_ext_codons = zlib.compress(ext_codon_bytes, level=1)

        self._write_varint(buffer, len(ext_len_bytes))
        buffer.write(ext_len_bytes)

        self._write_varint(buffer, len(pid_delta_bytes))
        self._write_varint(buffer, len(compressed_pid_deltas))
        buffer.write(compressed_pid_deltas)

        self._write_varint(buffer, len(ext_codon_bytes))
        self._write_varint(buffer, len(compressed_ext_codons))
        buffer.write(compressed_ext_codons)

    def _serialize(self, encoded_records: Sequence[dict], compact_mode: bool, use_token_entropy: bool) -> bytes:
        buffer = BytesIO()
        buffer.write(MAGIC)
        if compact_mode and use_token_entropy:
            mode = STREAM_COMPACT_EXACT_ENTROPY
        elif compact_mode:
            mode = STREAM_COMPACT_EXACT
        else:
            mode = STREAM_EXTENDED
        buffer.write(bytes([mode]))
        self._write_varint(buffer, len(encoded_records))
        for record in encoded_records:
            self._write_varint(buffer, record["original_index"])
            self._write_varint(buffer, record["sequence_length"])
            self._write_string(buffer, record["suffix"])
            tokens: List[Token] = record["tokens"]
            self._write_varint(buffer, len(tokens))
            if compact_mode and use_token_entropy:
                self._serialize_compact_entropy_record(buffer, tokens)
            elif compact_mode:
                for token in tokens:
                    self._write_varint(buffer, token.phrase_id)
                    self._write_varint(buffer, len(token.correction_list))
                    for extension_codon in token.correction_list:
                        buffer.write(bytes([CODON_TO_INDEX[extension_codon]]))
            else:
                for token in tokens:
                    buffer.write(bytes([TOKEN_CODES[token.token_type]]))
                    self._write_varint(buffer, token.phrase_id)
                    buffer.write(bytes([CODON_TO_INDEX[token.codon] if token.codon else 255]))
                    self._write_varint(buffer, len(token.correction_list))
                    for correction in token.correction_list:
                        buffer.write(bytes([CODON_TO_INDEX[correction]]))
        return buffer.getvalue()


class VCSDDecoder:
    """Main VCSD+ decoder."""

    def __init__(self) -> None:
        self.dictionary = Dictionary()

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

    def _read_string(self, buffer: BytesIO) -> str:
        length = self._read_varint(buffer)
        data = buffer.read(length)
        if len(data) != length:
            raise EOFError("Unexpected end of stream")
        return data.decode("ascii")

    def _read_codon_byte(self, buffer: BytesIO) -> str:
        byte = buffer.read(1)
        if not byte:
            raise EOFError("Unexpected end of stream")
        value = byte[0]
        if value == 255:
            return ""
        return INDEX_TO_CODON[value]

    def _read_exact(self, buffer: BytesIO, size: int) -> bytes:
        data = buffer.read(size)
        if len(data) != size:
            raise EOFError("Unexpected end of stream")
        return data

    def _decode_token(self, token_type: str, phrase_id: int, codon: str, correction_list: List[str]) -> Tuple[str, List[str]]:
        if token_type == "EXT":
            # EXT-run compaction packs the unmatched run into correction_list
            # and stores only the first codon in `codon` for legacy compatibility.
            # Prefer correction_list when present.
            if correction_list:
                phrase = list(correction_list)
            elif codon:
                phrase = [codon]
            else:
                phrase = []
        else:
            phrase = list(correction_list) if correction_list else list(self.dictionary.get_phrase(phrase_id))
            if codon:
                phrase.append(codon)
        return "".join(phrase), phrase

    def decode(self, bitstream: bytes) -> List[SequenceRecord]:
        buffer = BytesIO(bitstream)
        magic = buffer.read(len(MAGIC))
        if magic != MAGIC:
            raise ValueError("Invalid VCSD+ bitstream")
        mode_byte = buffer.read(1)
        if not mode_byte:
            raise EOFError("Unexpected end of stream")
        stream_mode = mode_byte[0]
        compact_mode = stream_mode in (STREAM_COMPACT_EXACT, STREAM_COMPACT_EXACT_ENTROPY)
        entropy_compact_mode = stream_mode == STREAM_COMPACT_EXACT_ENTROPY

        record_count = self._read_varint(buffer)
        records: List[Tuple[int, str]] = []
        for _ in range(record_count):
            original_index = self._read_varint(buffer)
            sequence_length = self._read_varint(buffer)
            suffix = self._read_string(buffer)
            token_count = self._read_varint(buffer)
            codons: List[str] = []

            if compact_mode and entropy_compact_mode:
                ext_len_blob_size = self._read_varint(buffer)
                ext_len_blob = self._read_exact(buffer, ext_len_blob_size)

                ext_len_reader = BytesIO(ext_len_blob)
                extension_lengths: List[int] = []
                for _ in range(token_count):
                    extension_lengths.append(self._read_varint(ext_len_reader))

                pid_raw_size = self._read_varint(buffer)
                pid_compressed_size = self._read_varint(buffer)
                pid_compressed = self._read_exact(buffer, pid_compressed_size)
                pid_raw = zlib.decompress(pid_compressed)
                if len(pid_raw) != pid_raw_size:
                    raise ValueError("Corrupt entropy block for phrase-id deltas")

                pid_reader = BytesIO(pid_raw)
                phrase_ids: List[int] = []
                previous_phrase_id = 0
                for _ in range(token_count):
                    delta_zz = self._read_varint(pid_reader)
                    delta = _zigzag_decode(delta_zz)
                    phrase_id = previous_phrase_id + delta
                    phrase_ids.append(phrase_id)
                    previous_phrase_id = phrase_id

                ext_raw_size = self._read_varint(buffer)
                ext_compressed_size = self._read_varint(buffer)
                ext_compressed = self._read_exact(buffer, ext_compressed_size)
                ext_raw = zlib.decompress(ext_compressed)
                if len(ext_raw) != ext_raw_size:
                    raise ValueError("Corrupt entropy block for extension codons")

                ext_cursor = 0
                for phrase_id, extension_len in zip(phrase_ids, extension_lengths):
                    next_cursor = ext_cursor + extension_len
                    if next_cursor > len(ext_raw):
                        raise ValueError("Corrupt extension payload")
                    extension_run = [INDEX_TO_CODON[value] for value in ext_raw[ext_cursor:next_cursor]]
                    ext_cursor = next_cursor
                    if phrase_id == 0:
                        if not extension_run:
                            raise ValueError("Invalid compact EXT token")
                        codons.extend(extension_run)
                        self.dictionary.add_phrase(extension_run)
                    else:
                        matched_phrase = list(self.dictionary.get_phrase(phrase_id))
                        codons.extend(matched_phrase)
                        codons.extend(extension_run)
                        if extension_run:
                            self.dictionary.add_phrase(matched_phrase + extension_run)

                if ext_cursor != len(ext_raw):
                    raise ValueError("Corrupt extension payload length")
            elif compact_mode:
                for _ in range(token_count):
                    phrase_id = self._read_varint(buffer)
                    extension_len = self._read_varint(buffer)
                    extension_run = [self._read_codon_byte(buffer) for _ in range(extension_len)]
                    if phrase_id == 0:
                        if not extension_run:
                            raise ValueError("Invalid compact EXT token")
                        codons.extend(extension_run)
                        self.dictionary.add_phrase(extension_run)
                    else:
                        matched_phrase = list(self.dictionary.get_phrase(phrase_id))
                        codons.extend(matched_phrase)
                        codons.extend(extension_run)
                        if extension_run:
                            self.dictionary.add_phrase(matched_phrase + extension_run)
            else:
                for _ in range(token_count):
                    token_type = TOKEN_NAMES[buffer.read(1)[0]]
                    phrase_id = self._read_varint(buffer)
                    codon = self._read_codon_byte(buffer)
                    correction_count = self._read_varint(buffer)
                    correction_list = [self._read_codon_byte(buffer) for _ in range(correction_count)]

                    if token_type == "EXT":
                        # EXT-run compaction: prefer correction_list; fall back
                        # to the single codon for legacy/short EXT cases.
                        if correction_list:
                            phrase_codons = list(correction_list)
                        elif codon:
                            phrase_codons = [codon]
                        else:
                            phrase_codons = []
                        if phrase_codons:
                            self.dictionary.add_phrase(phrase_codons)
                            codons.extend(phrase_codons)
                    elif token_type == "REF":
                        # REF: matched phrase from dictionary; correction_list
                        # carries the optional extension run.
                        if 0 < phrase_id <= len(self.dictionary):
                            matched_phrase = list(self.dictionary.get_phrase(phrase_id))
                        else:
                            matched_phrase = []
                        codons.extend(matched_phrase)
                        if correction_list:
                            codons.extend(correction_list)
                            extension_list = list(correction_list)
                            if matched_phrase:
                                self.dictionary.add_phrase(matched_phrase + extension_list)
                            else:
                                self.dictionary.add_phrase(extension_list)
                    else:
                        # AREF / RCREF / AEXT / RCEXT: correction_list IS the
                        # emitted codons (i.e., the codons that should replace
                        # the approximately-matched or RC-matched phrase).
                        if correction_list:
                            actual = list(correction_list)
                        elif 0 < phrase_id <= len(self.dictionary):
                            actual = list(self.dictionary.get_phrase(phrase_id))
                        else:
                            actual = []
                        codons.extend(actual)
                        if actual:
                            self.dictionary.add_phrase(actual)

            sequence = "".join(codons) + suffix
            if len(sequence) != sequence_length:
                raise ValueError(
                    f"VCSD+ decode produced {len(sequence)} bases, expected {sequence_length}"
                )
            if suffix:
                self.dictionary.add_phrase([suffix])
            records.append((original_index, sequence))

        records.sort(key=lambda item: item[0])
        return [(f"sequence_{index + 1}", sequence) for index, (_, sequence) in enumerate(records)]


def encode_dataset(
    sequences: Union[str, Sequence[SequenceRecord]],
    use_approximate: bool = False,
    use_reverse_complement: bool = False,
    sequence_ordering: bool = False,
    use_token_entropy: bool = False,
) -> bytes:
    encoder = VCSDEncoder(
        use_approximate=use_approximate,
        use_reverse_complement=use_reverse_complement,
        sequence_ordering=sequence_ordering,
        use_token_entropy=use_token_entropy,
    )
    return encoder.encode(sequences)


def decode_dataset(bitstream: bytes) -> List[SequenceRecord]:
    return VCSDDecoder().decode(bitstream)


__all__ = [
    "CodonSequence",
    "Dictionary",
    "Token",
    "VCSDEncoder",
    "VCSDDecoder",
    "CompressionMetrics",
    "encode_dataset",
    "decode_dataset",
]