"""Small retry/backoff helper (PRD Section 10.1: "retry/resume logic, given free-tier
bandwidth and session limits"). No extra dependency — exponential backoff with jitter.
"""

from __future__ import annotations

import random
import time
from typing import Callable, TypeVar

T = TypeVar("T")


def retry(
    fn: Callable[[], T],
    *,
    attempts: int = 4,
    base_delay_s: float = 2.0,
    max_delay_s: float = 60.0,
    on_retry: Callable[[int, Exception], None] | None = None,
) -> T:
    """Call fn(), retrying on any Exception with exponential backoff + jitter."""
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - deliberately broad: network/API errors vary by provider
            last_exc = exc
            if attempt == attempts:
                break
            delay = min(max_delay_s, base_delay_s * (2 ** (attempt - 1))) * (0.5 + random.random())
            if on_retry:
                on_retry(attempt, exc)
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc
