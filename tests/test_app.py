from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from sc_copilot.data_gen import generate

APP = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"

def test_app_demo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    generate(tmp_path / "data" / "sc.db")
    at = AppTest.from_file(str(APP)).run(timeout=30)
    assert not at.exception
    at.text_input[0].set_value("Quel est le taux OTIF ?").run(timeout=30)
    assert not at.exception and not at.error
    assert any("73.6%" in m.value for m in at.markdown)
