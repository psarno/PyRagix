from __future__ import annotations

import sys
import threading
from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional


class IngestionStage(str, Enum):
    """Enumeration of ingestion stages for progress reporting."""

    SCANNING = "scanning"
    RESETTING = "resetting"
    DISCOVERY = "discovery"
    FILE_STARTED = "file_started"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    FILE_SKIPPED = "file_skipped"
    FILE_COMPLETED = "file_completed"
    PERSISTING = "persisting"
    COMPLETED = "completed"
    ERROR = "error"

    @property
    def default_message(self) -> str:
        """Fallback string for stages when no explicit message provided."""
        defaults: dict[IngestionStage, str] = {
            IngestionStage.SCANNING: "Scanning documents...",
            IngestionStage.RESETTING: "Resetting indexes...",
            IngestionStage.DISCOVERY: "Preparing worklist...",
            IngestionStage.FILE_STARTED: "Processing document...",
            IngestionStage.CHUNKING: "Chunking document...",
            IngestionStage.EMBEDDING: "Embedding chunks...",
            IngestionStage.INDEXING: "Indexing chunks...",
            IngestionStage.FILE_SKIPPED: "Skipping document...",
            IngestionStage.FILE_COMPLETED: "Document completed.",
            IngestionStage.PERSISTING: "Persisting indexes...",
            IngestionStage.COMPLETED: "Ingestion complete.",
            IngestionStage.ERROR: "Error detected.",
        }
        return defaults.get(self, "Working...")


@dataclass
class _ProgressSnapshot:
    """Mutable status snapshot rendered by the spinner."""

    stage: IngestionStage = IngestionStage.SCANNING
    message: str = "Preparing ingestion..."
    total_files: Optional[int] = None
    files_completed: int = 0
    chunk_total: int = 0
    skipped_already_processed: int = 0
    skipped_problems: int = 0


class ConsoleSpinnerProgress:
    """Renders a lightweight spinner with status updates during ingestion."""

    def __init__(self, *, enabled: Optional[bool] = None, interval: float = 0.12) -> None:
        super().__init__()
        self._enabled = sys.stdout.isatty() if enabled is None else enabled
        self._interval = max(interval, 0.05)
        self._frames = ["-", "\\", "|", "/"]
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._snapshot = _ProgressSnapshot()
        self._render_thread: threading.Thread | None = None
        self._last_line_length = 0
        self._active = False

    @property
    def enabled(self) -> bool:
        """Return True when spinner output is active."""
        return self._enabled

    @property
    def is_active(self) -> bool:
        """Return True if the render loop is currently running."""
        return self._active and self._render_thread is not None

    def start(
        self,
        *,
        stage: IngestionStage | None = None,
        message: str | None = None,
        total_files: Optional[int] = None,
    ) -> None:
        """Begin rendering the spinner, updating initial state if provided."""
        if not self._enabled:
            return

        with self._lock:
            if stage is not None:
                self._snapshot.stage = stage
            if message is not None:
                self._snapshot.message = message
            elif stage is not None:
                self._snapshot.message = stage.default_message
            if total_files is not None:
                self._snapshot.total_files = total_files
            already_active = self._active

        if already_active:
            return

        self._stop_event.clear()
        self._render_thread = threading.Thread(
            target=self._render_loop, daemon=True, name="ingestion-spinner"
        )
        self._render_thread.start()
        self._active = True

    def update(
        self,
        *,
        stage: IngestionStage | None = None,
        message: str | None = None,
        total_files: Optional[int] = None,
        files_completed: Optional[int] = None,
        chunk_total: Optional[int] = None,
        skipped_already_processed: Optional[int] = None,
        skipped_problems: Optional[int] = None,
    ) -> None:
        """Update the current snapshot used by the spinner."""
        if not self._enabled:
            return

        with self._lock:
            if stage is not None:
                self._snapshot.stage = stage
                if message is None:
                    message = stage.default_message
            if message is not None:
                self._snapshot.message = message
            if total_files is not None:
                self._snapshot.total_files = total_files
            if files_completed is not None:
                self._snapshot.files_completed = max(files_completed, 0)
            if chunk_total is not None:
                self._snapshot.chunk_total = max(chunk_total, 0)
            if skipped_already_processed is not None:
                self._snapshot.skipped_already_processed = max(skipped_already_processed, 0)
            if skipped_problems is not None:
                self._snapshot.skipped_problems = max(skipped_problems, 0)

    def write_line(self, text: str) -> None:
        """Print a full line while preserving spinner output."""
        if not self._enabled:
            print(text)
            return

        with self._lock:
            self._clear_line_locked()
            print(text)
            _ = sys.stdout.flush()

    def stop(self, final_message: str | None = None) -> None:
        """Terminate the render loop and optionally print a final message."""
        if not self._enabled:
            if final_message:
                print(final_message)
            return

        if not self._active:
            if final_message:
                print(final_message)
            return

        self._stop_event.set()
        if self._render_thread is not None:
            self._render_thread.join()
        self._render_thread = None
        self._active = False

        with self._lock:
            self._clear_line_locked()

        if final_message:
            print(final_message)
            _ = sys.stdout.flush()

    def _render_loop(self) -> None:
        frame_index = 0
        while not self._stop_event.is_set():
            with self._lock:
                snapshot = replace(self._snapshot)
            frame = self._frames[frame_index % len(self._frames)]
            frame_index += 1
            line = self._build_line(frame, snapshot)
            self._write_inline(line)
            if self._stop_event.wait(self._interval):
                break

    def _build_line(self, frame: str, snapshot: _ProgressSnapshot) -> str:
        parts: list[str] = [frame]

        if snapshot.total_files:
            total = max(snapshot.total_files, 0)
            completed = min(max(snapshot.files_completed, 0), total)
            parts.append(f"[{completed}/{total}]")

        parts.append(snapshot.message.strip())

        extras: list[str] = []
        if snapshot.chunk_total:
            extras.append(f"chunks={snapshot.chunk_total}")
        if snapshot.skipped_already_processed:
            extras.append(f"skipped={snapshot.skipped_already_processed}")
        if snapshot.skipped_problems:
            extras.append(f"errors={snapshot.skipped_problems}")

        if extras:
            parts.append("(" + " | ".join(extras) + ")")

        return " ".join(parts)

    def _write_inline(self, line: str) -> None:
        if not self._enabled:
            return

        with self._lock:
            _ = sys.stdout.write("\r")
            _ = sys.stdout.write(line)
            pad = self._last_line_length - len(line)
            if pad > 0:
                _ = sys.stdout.write(" " * pad)
                _ = sys.stdout.write("\r")
                _ = sys.stdout.write(line)
            _ = sys.stdout.flush()
            self._last_line_length = len(line)

    def _clear_line_locked(self) -> None:
        if self._last_line_length <= 0:
            return
        _ = sys.stdout.write("\r")
        _ = sys.stdout.write(" " * self._last_line_length)
        _ = sys.stdout.write("\r")
        _ = sys.stdout.flush()
        self._last_line_length = 0


__all__ = ["ConsoleSpinnerProgress", "IngestionStage"]
