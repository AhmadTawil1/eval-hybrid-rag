import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Callable


class SlidingWindowLimiter:
    """At most `limit` hits per key within the last `window_seconds`. In memory, per process."""

    def __init__(self, limit: int, window_seconds: float, clock: Callable[[], float] = time.monotonic):
        self.limit = limit
        self.window = window_seconds
        self.clock = clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, int]:
        """Record a hit if allowed. Returns (allowed, seconds_until_a_slot_frees_up)."""
        now = self.clock()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return False, max(1, int(hits[0] + self.window - now) + 1)
            hits.append(now)
            return True, 0


class DailyBudget:
    """A count of generated answers per UTC day. In memory: a restart resets it, so the
    hard spending limit on the OpenAI key stays the real backstop."""

    def __init__(self, limit: int, today: Callable[[], str] | None = None):
        self.limit = limit
        self._today = today or (lambda: datetime.now(timezone.utc).date().isoformat())
        self._day = self._today()
        self._used = 0
        self._lock = threading.Lock()

    def _roll_over(self) -> None:
        today = self._today()
        if today != self._day:
            self._day, self._used = today, 0

    def remaining(self) -> int:
        with self._lock:
            self._roll_over()
            return max(0, self.limit - self._used)

    def try_spend(self) -> bool:
        with self._lock:
            self._roll_over()
            if self._used >= self.limit:
                return False
            self._used += 1
            return True
