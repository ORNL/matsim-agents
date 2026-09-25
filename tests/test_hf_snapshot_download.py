from __future__ import annotations

import importlib.util
import io
from pathlib import Path
from types import ModuleType

import pytest


def _load_downloader() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "deployments/common/download_hf_snapshot.py"
    spec = importlib.util.spec_from_file_location("download_hf_snapshot", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_download_file_writes_atomically(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    downloader = _load_downloader()
    requests = []

    def open_request(request):
        requests.append(request)
        return io.BytesIO(b"model data")

    monkeypatch.setattr(downloader.urllib.request, "urlopen", open_request)
    destination = tmp_path / "nested/model.safetensors"

    downloader.download_file("org/model", "model.safetensors", destination, "secret")

    assert destination.read_bytes() == b"model data"
    assert not list(tmp_path.rglob("*.part.*"))
    assert requests[0].get_header("Authorization") == "Bearer secret"


def test_download_file_skips_existing_nonempty_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    downloader = _load_downloader()
    destination = tmp_path / "config.json"
    destination.write_bytes(b"complete")

    def unexpected_request(_request):
        raise AssertionError("completed files must not be downloaded again")

    monkeypatch.setattr(downloader.urllib.request, "urlopen", unexpected_request)

    downloader.download_file("org/model", "config.json", destination, None)

    assert destination.read_bytes() == b"complete"


def test_download_file_removes_partial_file_after_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    downloader = _load_downloader()

    class BrokenResponse(io.BytesIO):
        def __init__(self) -> None:
            super().__init__(b"partial")
            self.read_count = 0

        def read(self, size: int = -1) -> bytes:
            self.read_count += 1
            if self.read_count > 1:
                raise OSError("connection lost")
            return super().read(size)

    monkeypatch.setattr(downloader.urllib.request, "urlopen", lambda _request: BrokenResponse())
    destination = tmp_path / "model.safetensors"

    with pytest.raises(OSError, match="connection lost"):
        downloader.download_file("org/model", "model.safetensors", destination, None)

    assert not destination.exists()
    assert not list(tmp_path.glob("*.part.*"))
