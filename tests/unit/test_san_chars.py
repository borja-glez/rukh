"""Tests for rukh.tokenize.san_chars: alphabet, round trip and UCI to SAN conversion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rukh.tokenize.san_chars import ALPHABET, SanCharTokenizer, san_text, uci_to_san

pytestmark = pytest.mark.unit


def test_alphabet_has_32_symbols_and_specials_first() -> None:
    tok = SanCharTokenizer()
    assert len(ALPHABET) == 32
    assert len(set(ALPHABET)) == 32
    assert len(tok) == 35
    assert tok.ids[:3] == ["<pad>", "<bos>", "<eos>"]
    assert tok.vocab[" "] == 3
    assert tok.vocab["/"] == 34


def test_round_trip() -> None:
    tok = SanCharTokenizer()
    text = san_text(
        "1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 4.Bxc6 dxc6 5.O-O f6 6.d4 exd4 7.Nxd4 c5", "1/2-1/2"
    )
    ids = tok.encode(text)
    assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id
    assert len(ids) == len(text) + 2
    assert tok.decode(ids) == text


def test_rejects_unknown_character() -> None:
    with pytest.raises(ValueError):
        SanCharTokenizer().encode("1.e4 e5 {comment}")


def test_uci_to_san_numbers_and_notation() -> None:
    san = uci_to_san("e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5c6 d7c6 e1g1 f7f6")
    assert san == "1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 4.Bxc6 dxc6 5.O-O f6"
    assert uci_to_san("e2e4 d7d5 e4d5 d8d5 b1c3 d5e5") == "1.e4 d5 2.exd5 Qxd5 3.Nc3 Qe5+"
    with pytest.raises(ValueError):
        uci_to_san("e2e5")


def test_fixture_san_ids_match(repo_root: Path) -> None:
    tok = SanCharTokenizer()
    fixture = json.loads(
        (repo_root / "artifacts" / "tokenizer" / "fixtures" / "games.json").read_text("utf-8")
    )
    for entry in fixture:
        assert entry["san_ids"] == tok.encode(san_text(entry["san"], entry["result"]))
        assert uci_to_san(entry["uci"]) == entry["san"]
