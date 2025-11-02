"""Minimal pytest stub for type checking.

This module is only intended to satisfy static analysis. It does not provide
runtime behavior. Real test execution should rely on the actual pytest
package.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar, overload

_T = TypeVar("_T")


class MonkeyPatch:
    def setattr(self, *args: Any, **kwargs: Any) -> None: ...
    def setenv(self, *args: Any, **kwargs: Any) -> None: ...
    def delattr(self, *args: Any, **kwargs: Any) -> None: ...
    def undo(self) -> None: ...


class CaptureResult:
    out: str = ""
    err: str = ""


class CaptureFixture:
    def read(self) -> str: ...
    def readouterr(self) -> CaptureResult: ...
    def flush(self) -> None: ...


def fixture(func: Callable[..., _T]) -> Callable[..., _T]: ...


@overload
def raises(expected_exception: type[BaseException]) -> Any: ...


@overload
def raises(expected_exception: type[BaseException], match: str) -> Any: ...


def raises(*args: Any, **kwargs: Any) -> Any: ...


def mark(*args: Any, **kwargs: Any) -> Any: ...


__all__ = [
    "MonkeyPatch",
    "CaptureFixture",
    "CaptureResult",
    "fixture",
    "raises",
    "mark",
]
