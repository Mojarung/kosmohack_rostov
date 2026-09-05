"""Настоящий API и данные кейса; только набор пользовательских отчётов изолирован фикстурой."""
import tempfile
from pathlib import Path
import uvicorn
from service import field_store, polygons
from tests.e2e.anomalies_fixtures import seed_saved

if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="anomalies-e2e-") as folder:
        polygons.POLYGONS_DIR = Path(folder) / "polygons"
        field_store.ROOT = Path(folder) / "fields"
        seed_saved()
        uvicorn.run("service.app:app", host="127.0.0.1", port=8015)
