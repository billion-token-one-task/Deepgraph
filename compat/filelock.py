"""Best-effort cross-process file locks (POSIX + Windows).

DeepGraph uses flock(2) on Linux for "singleton" style locks in a few long-running
components. Windows doesn't ship fcntl, so we use msvcrt byte-range locks as a
reasonable approximation for local development machines.
"""

from __future__ import annotations

import os
from types import TracebackType
from typing import Optional, Type


class FileLock:
    def __init__(self, path: str) -> None:
        self._path = path
        self._handle = None

    def acquire(self) -> None:
        if self._handle is not None:
            return

        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        self._handle = open(self._path, "a+", encoding="utf-8")

        if os.name == "nt":
            import msvcrt

            self._handle.seek(0)
            # Lock first byte of the file (shared lock storage may be empty).
            msvcrt.locking(self._handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)

    def release(self) -> None:
        if self._handle is None:
            return

        if os.name == "nt":
            import msvcrt

            try:
                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        else:
            import fcntl

            try:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass

        try:
            self._handle.close()
        finally:
            self._handle = None

    def try_acquire(self) -> bool:
        if self._handle is not None:
            return True

        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        handle = open(self._path, "a+", encoding="utf-8")
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError:
                    handle.close()
                    return False
            else:
                import fcntl

                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    handle.close()
                    return False
        except OSError:
            try:
                handle.close()
            finally:
                return False

        self._handle = handle
        return True

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self.release()
