import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def temporary_workdir():
    root = Path(".test_runs").resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"run_{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
