"""
Token bucket rate limiter to stay within Gemini free tier (5 RPM).
With combined_analysis_node, each query uses 2 LLM calls max,
so this limiter is a safety net, not the primary throttle.
"""

import time
import threading
from collections import deque


class RateLimiter:
    def __init__(self, max_calls: int = 4, window_seconds: float = 62.0):
        self.max_calls = max_calls
        self.window = window_seconds
        self._timestamps: deque = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            while self._timestamps and now - self._timestamps[0] > self.window:
                self._timestamps.popleft()

            if len(self._timestamps) >= self.max_calls:
                oldest = self._timestamps[0]
                sleep_for = self.window - (now - oldest) + 0.2
                if sleep_for > 0:
                    time.sleep(sleep_for)
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] > self.window:
                    self._timestamps.popleft()

            self._timestamps.append(time.monotonic())


_limiter = RateLimiter(max_calls=4, window_seconds=62.0)


def throttle() -> None:
    """Call this before every LLM invocation."""
    _limiter.acquire()