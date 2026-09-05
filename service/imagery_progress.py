"""Прогресс карты на диске: доступен после перезагрузки страницы и из процесса сборщика."""
import datetime as dt
import os
import threading
from functools import wraps

from service.field_store import _write

ACTIVE = {"queued", "catalog", "download", "render"}
_lock = threading.Lock()
_jobs = {}


def _path(pid, year):
    from service.imagery import directory
    return directory(pid, year) / "progress.json"


def read(pid, year):
    import json
    path = _path(pid, year)
    if not path.exists():
        return None
    state = json.loads(path.read_text(encoding="utf-8"))
    if state["stage"] in ACTIVE and os.name == "posix":
        try:
            os.kill(state["owner"], 0)
        except ProcessLookupError:
            state.update(stage="error", message="Сбор прерван перезапуском сервера. Запустите его снова.")
        except PermissionError:
            pass
    return {key: value for key, value in state.items() if key != "owner"}


class Progress:
    def __init__(self, pid, year, fresh=False):
        self.path = _path(pid, year)
        self.state = (None if fresh else read(pid, year)) or {"started": dt.datetime.now(dt.UTC).isoformat()}

    def update(self, stage, **values):
        self.state.update(stage=stage, owner=os.getpid(), updated=dt.datetime.now(dt.UTC).isoformat(), **values)
        _write(self.path, self.state)


def tracked(collect):
    @wraps(collect)
    def run(pid, geometry, year):
        progress = Progress(pid, year)
        progress.update("catalog", processed=0, total=0, suitable=0, message=None)
        try:
            result = collect(pid, geometry, year)
        except Exception as exc:
            Progress(pid, year).update("error", message=f"Не удалось собрать снимки: {exc}")
            raise
        Progress(pid, year).update("done", processed=len(result["scenes"]), total=len(result["scenes"]), suitable=len(result["scenes"]))
        return result
    return run


def submit(pool, pid, geometry, year):
    """Повторный запрос в этом сервере присоединяется к существующему сбору."""
    from service.imagery import collect
    key = (pid, year)
    with _lock:
        current = _jobs.get(key)
        if current is not None and not current.done():
            return current
        progress = Progress(pid, year, fresh=True)
        progress.update("queued", processed=0, total=0, suitable=0, message=None)
        try:
            future = pool.submit(collect, pid, geometry, year)
        except Exception as exc:
            progress.update("error", message=f"Не удалось запустить сбор: {exc}")
            raise
        _jobs[key] = future
    def finished(result):
        with _lock:
            if _jobs.get(key) is not result:
                return
            _jobs.pop(key, None)
            if result.cancelled() or result.exception() is not None:
                state = read(pid, year)
                if not state or state["stage"] != "error":
                    Progress(pid, year).update("error", message="Процесс сбора прерван. Повторите загрузку снимков.")
    future.add_done_callback(finished)
    return future
