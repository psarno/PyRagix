"""Simple terminal spinner for long-running operations."""

from __future__ import annotations

import itertools
import sys
import threading
import time
from contextlib import AbstractContextManager
from typing import Optional


class Spinner(AbstractContextManager["Spinner"]):
    """A lightweight CLI spinner that runs in a background thread."""

    def __init__(
        self,
        message: str = "",
        interval: float = 0.1,
        enabled: bool = True,
        final_message: Optional[str] = None,
    ) -> None:
        self.message = message
        self.interval = interval
        self.enabled = enabled
        self.final_message = final_message
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._line_width = len(message) + 2

    def __enter__(self) -> "Spinner":
        if self.enabled:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self.enabled:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        self._clear_line()
        if self.final_message:
            sys.stdout.write(f"{self.final_message}\n")
            sys.stdout.flush()

    def _run(self) -> None:
        spinner_cycle = itertools.cycle("|/-\\")
        while not self._stop_event.is_set():
            symbol = next(spinner_cycle)
            frame = f"{self.message} {symbol}".rstrip()
            self._line_width = max(self._line_width, len(frame))
            sys.stdout.write(f"\r{frame}")
            sys.stdout.flush()
            time.sleep(self.interval)
        self._clear_line()

    @staticmethod
    def _clear_line_width(width: int) -> None:
        sys.stdout.write("\r" + (" " * width) + "\r")
        sys.stdout.flush()

    def _clear_line(self) -> None:
        self._clear_line_width(self._line_width)


__all__ = ["Spinner"]
