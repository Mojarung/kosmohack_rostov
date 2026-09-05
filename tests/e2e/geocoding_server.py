"""Настоящий FastAPI с локальной имитацией только внешнего Nominatim для воспроизводимого e2e."""
import json
import os
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import uvicorn


class Upstream(BaseHTTPRequestHandler):
    failures = 0

    def do_GET(self):
        query = parse_qs(urlparse(self.path).query).get("q", [""])[0]
        if query == "Медленный ответ":
            time.sleep(2)
        if query == "Ошибка источника" and Upstream.failures == 0:
            Upstream.failures += 1
            self.send_response(503)
            self.end_headers()
            return
        results = [] if query == "Несуществующее место" else [
            {"osm_type": "way", "osm_id": 1, "display_name": "Большая Садовая улица, Ростов-на-Дону, Ростовская область",
             "lat": "47.22", "lon": "39.72", "boundingbox": ["47.21", "47.23", "39.70", "39.74"],
             "address": {"road": "Большая Садовая улица", "state": "Ростовская область"}},
            {"osm_type": "relation", "osm_id": 2, "display_name": "Ростовская область, Россия",
             "lat": "47.5", "lon": "41.0", "boundingbox": ["45.95", "50.22", "38.22", "44.33"],
             "address": {"state": "Ростовская область"}},
        ]
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(results).encode())

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="geocoding-e2e-") as cache:
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        threading.Thread(target=upstream.serve_forever, daemon=True).start()
        os.environ["NOMINATIM_URL"] = f"http://127.0.0.1:{upstream.server_port}/search"
        os.environ["GEOCODING_CACHE"] = f"{cache}/cache.sqlite"
        uvicorn.run("service.app:app", host="127.0.0.1", port=int(os.environ.get("E2E_PORT", "8011")))
