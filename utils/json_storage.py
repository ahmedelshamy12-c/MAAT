import json
import os
import tempfile
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


class StorageError(RuntimeError):
    pass


@contextmanager
def file_lock(path: Path):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.touch(exist_ok=True)
    handle = lock_path.open("r+b")
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def ensure_json_file(path: Path, default):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        write_json(path, default)


def _atomic_write_unlocked(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.stem}-",
        suffix=".tmp",
        text=True,
    )
    with os.fdopen(fd, "w", encoding="utf-8") as tmp:
        json.dump(payload, tmp, ensure_ascii=False, indent=2)
        tmp.write("\n")
    os.replace(tmp_name, path)


def read_json(path: Path, default):
    ensure_json_file(path, default)
    with file_lock(path):
        try:
            content = path.read_text(encoding="utf-8")
            if not content.strip():
                return deepcopy(default)
            return json.loads(content)
        except json.JSONDecodeError as error:
            backup = path.with_suffix(path.suffix + ".corrupt")
            path.replace(backup)
            _atomic_write_unlocked(path, default)
            raise StorageError(f"Malformed JSON moved to {backup.name}") from error
        except OSError as error:
            raise StorageError(f"Could not read {path.name}: {error}") from error


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(path):
        try:
            _atomic_write_unlocked(path, payload)
        except OSError as error:
            raise StorageError(f"Could not write {path.name}: {error}") from error


def update_json(path: Path, default, mutator):
    with file_lock(path):
        try:
            if not path.exists() or not path.read_text(encoding="utf-8").strip():
                data = deepcopy(default)
            else:
                data = json.loads(path.read_text(encoding="utf-8"))
            result = mutator(data)
            _atomic_write_unlocked(path, data)
            return result
        except json.JSONDecodeError as error:
            backup = path.with_suffix(path.suffix + ".corrupt")
            path.replace(backup)
            _atomic_write_unlocked(path, default)
            raise StorageError(f"Malformed JSON moved to {backup.name}") from error
        except OSError as error:
            raise StorageError(f"Could not update {path.name}: {error}") from error
