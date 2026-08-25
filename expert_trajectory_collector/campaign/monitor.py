"""Dependency-free progress monitor with pause/resume controls."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading

from .state import aggregate_status, set_paused


PAGE = """<!doctype html><html lang='zh-CN'><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>HNUTER Expert Collection Monitor</title>
<style>body{font:15px system-ui;background:#0d141b;color:#dbe7ef;max-width:760px;margin:50px auto;padding:0 20px}
.card{background:#14212b;border:1px solid #29404d;border-radius:12px;padding:24px}h1{font-size:22px}
.bar{height:14px;background:#263640;border-radius:8px;overflow:hidden}.fill{height:100%;background:#31c48d}
button{margin:16px 8px 0 0;padding:9px 16px;border:0;border-radius:7px;cursor:pointer}pre{white-space:pre-wrap}</style>
<div class='card'><h1>Expert Collection Monitor</h1><div id='summary'>loading</div>
<div class='bar'><div class='fill' id='fill'></div></div><button onclick="control('pause')">暂停</button>
<button onclick="control('resume')">继续</button><pre id='details'></pre></div>
<script>async function refresh(){let s=await(await fetch('/api/status')).json();
let p=s.path_target?100*s.path_accepted/s.path_target:0;fill.style.width=p.toFixed(1)+'%';
summary.textContent=`${s.state} · ${s.path_accepted}/${s.path_target} paths · ${s.environment_finished}/${s.environment_target} env`;
details.textContent=JSON.stringify(s,null,2)}async function control(a){await fetch('/api/'+a,{method:'POST'});refresh()}
refresh();setInterval(refresh,2000)</script></html>"""


def make_handler(root: Path) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/":
                self._reply(200, "text/html; charset=utf-8", PAGE.encode())
            elif self.path == "/api/status":
                body = json.dumps(aggregate_status(root), ensure_ascii=False).encode()
                self._reply(200, "application/json", body)
            else:
                self._reply(404, "text/plain", b"not found")

        def do_POST(self) -> None:  # noqa: N802
            if self.path not in {"/api/pause", "/api/resume"}:
                self._reply(404, "text/plain", b"not found")
                return
            set_paused(root, self.path.endswith("pause"))
            body = json.dumps(aggregate_status(root), ensure_ascii=False).encode()
            self._reply(200, "application/json", body)

        def log_message(self, *_: object) -> None:
            return

    return Handler


def start_monitor(root: Path, host: str, port: int) -> tuple[ThreadingHTTPServer, threading.Thread]:
    server = ThreadingHTTPServer((host, port), make_handler(root))
    thread = threading.Thread(target=server.serve_forever, name="campaign-monitor", daemon=True)
    thread.start()
    return server, thread
