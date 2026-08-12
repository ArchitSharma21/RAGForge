from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import gradio as gr

from ragforge.api import create_api
from ragforge.ui import build_ui

app = create_api()
ui = build_ui()
app = gr.mount_gradio_app(app, ui, path="/")
