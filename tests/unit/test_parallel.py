"""Tests for rukh.data.parallel: the feeder never runs ahead of the consumer."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest

from rukh.data.parallel import imap_bounded, run_batches
from rukh.data.uci import convert_batch

pytestmark = pytest.mark.unit


class _Immediate:
    def __init__(self, value: object) -> None:
        self.value = value

    def get(self) -> object:
        return self.value


class _FakePool:
    """Runs the job at submission time, so only the queue bound is under test."""

    def __init__(self) -> None:
        self.submitted = 0

    def apply_async(self, func: Callable[..., object], args: tuple[object, ...]) -> _Immediate:
        self.submitted += 1
        return _Immediate(func(*args))


def _counted(n: int, seen: list[int]) -> Iterator[int]:
    for i in range(n):
        seen.append(i)
        yield i


def test_imap_bounded_keeps_the_queue_short() -> None:
    seen: list[int] = []
    pool = _FakePool()
    produced: list[int] = []
    for value in imap_bounded(pool, lambda x: x * 2, _counted(50, seen), max_in_flight=4):
        # The feeder is at most ``max_in_flight`` batches ahead of the consumer.
        assert len(seen) - len(produced) <= 4
        produced.append(value)
    assert produced == [i * 2 for i in range(50)]
    assert pool.submitted == 50
    with pytest.raises(ValueError):
        next(imap_bounded(pool, lambda x: x, [1], 0))


def _row(movetext: str) -> dict[str, object]:
    return {
        "Site": "https://lichess.org/x",
        "movetext": movetext,
        "WhiteElo": 1900,
        "BlackElo": 1850,
        "Result": "1-0",
        "TimeControl": "300+0",
        "UTCDate": "2025.01.01",
        "ECO": "C60",
        "month": "2025-01",
    }


def test_run_batches_bounds_a_real_spawn_pool() -> None:
    """With ``workers`` processes the generator stays within ``2 * workers + 1`` batches."""
    workers = 2
    materialised: list[int] = []

    def jobs() -> Iterator[tuple[list[dict[str, object]], int, int]]:
        for index in range(12):
            materialised.append(index)
            yield [_row("1. e4 e5 2. Nf3 Nc6")], 0, 300

    consumed = 0
    for result in run_batches(convert_batch, jobs(), workers):
        consumed += 1
        assert len(materialised) - consumed <= 2 * workers + 1
        assert result["counts"]["kept"] == 1  # type: ignore[index]
    assert consumed == 12


def test_run_batches_is_serial_with_one_worker() -> None:
    assert list(run_batches(lambda x: x + 1, [1, 2, 3], workers=1)) == [2, 3, 4]
