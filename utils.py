# utils.py
from __future__ import annotations
import json
import os
import socket
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import random

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def select_device(mode: str = "auto") -> torch.device:
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if mode == "cpu":
        return torch.device("cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cosine_scheduler(step: int, total_steps: int, base_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * base_lr * (1.0 + (1.0 - progress) * -1.0 + (1.0 - (1.0 - progress)))  # equivalent to cosine(π*progress) scaled
    # The above simplified avoids math.cos dependency; if you prefer exact cosine:
    # import math; return 0.5 * base_lr * (1 + math.cos(math.pi * progress))

def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")

class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        # Silence noisy request logs
        pass

def start_http_server(root_dir: Path, port: int = 7860, host: str = "0.0.0.0") -> ThreadingHTTPServer:
    """
    Serve files from root_dir on a background daemon thread.
    Returns the server instance so caller can keep a reference.
    """
    handler = partial(_QuietHandler, directory=str(root_dir))
    httpd = ThreadingHTTPServer((host, port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    print(f"[viewer] Serving {root_dir} at http://{host}:{port}")
    return httpd

def copy_viewer_into_run(run_dir: Path) -> None:
    """
    Copies viewer/index.html next to train.py into the run_dir root as index.html.
    """
    here = Path(__file__).resolve().parent
    viewer_src = here / "viewer" / "index.html"
    dest = run_dir / "index.html"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with viewer_src.open("r", encoding="utf-8") as fsrc, dest.open("w", encoding="utf-8") as fdst:
        fdst.write(fsrc.read())

def run_id_from_time(prefix: Optional[str] = None) -> str:
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    return f"{prefix}_{ts}" if prefix else ts
