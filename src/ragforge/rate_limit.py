from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from .config import get_settings


class RateLimitExceeded(ValueError):
    pass


class SlidingWindowLimiter:
    def __init__(self):
        self.limit = get_settings().queries_per_hour_per_ip
        self.events: dict[str, deque[float]] = defaultdict(deque)
        self.lock = threading.Lock()

    def check(self, key: str) -> None:
        now = time.time()
        cutoff = now - 3600
        with self.lock:
            q = self.events[key]
            while q and q[0] < cutoff:
                q.popleft()
            if len(q) >= self.limit:
                raise RateLimitExceeded(f"Rate limit reached ({self.limit} operations/hour for this client).")
            q.append(now)


limiter = SlidingWindowLimiter()
