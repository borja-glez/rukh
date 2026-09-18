"""Bounded fan-out over a ``spawn`` process pool.

``multiprocessing.Pool.imap`` drains its input iterable as fast as it can, so feeding it a
generator of 20 000-game batches pulls a whole month into memory before the first result comes
back. ``run_batches`` keeps at most ``2 * workers`` batches in flight: a new job is submitted
only when an older one has been consumed, so the feeder advances at the speed of the writer.
"""

from __future__ import annotations

import multiprocessing
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from typing import Protocol


class AsyncResult[R](Protocol):
    """What ``imap_bounded`` needs from ``Pool.apply_async``."""

    def get(self) -> R: ...


class Pool(Protocol):
    """What ``imap_bounded`` needs from ``multiprocessing.Pool`` (tests use a fake)."""

    def apply_async[R](
        self, func: Callable[..., R], args: tuple[object, ...]
    ) -> AsyncResult[R]: ...


def imap_bounded[J, R](
    pool: Pool,
    func: Callable[[J], R],
    jobs: Iterable[J],
    max_in_flight: int,
) -> Iterator[R]:
    """``pool.imap`` with backpressure: never more than ``max_in_flight`` jobs are queued."""
    if max_in_flight < 1:
        raise ValueError("max_in_flight must be at least 1")
    pending: deque[AsyncResult[R]] = deque()
    for job in jobs:
        pending.append(pool.apply_async(func, (job,)))
        if len(pending) >= max_in_flight:
            yield pending.popleft().get()
    while pending:
        yield pending.popleft().get()


def run_batches[J, R](func: Callable[[J], R], jobs: Iterable[J], workers: int) -> Iterator[R]:
    """Map ``func`` over ``jobs``; serially with one worker, else in a bounded spawn pool.

    ``func`` and the jobs must be picklable: every worker entry point is a module-level
    function (Windows starts processes with ``spawn``).
    """
    if workers <= 1:
        yield from map(func, jobs)
        return
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(workers) as pool:
        yield from imap_bounded(pool, func, jobs, 2 * workers)
