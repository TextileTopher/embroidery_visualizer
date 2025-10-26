from pathlib import Path
from typing import Dict

import pytest
from fastapi.testclient import TestClient

import render_service
from render_service import MAX_INPUT_BYTES, app


@pytest.fixture
def client():
    return TestClient(app)


def _base_payload() -> Dict:
    return {
        "input": {
            "bucket_name": "digitizer",
            "file_path": "test",
            "file_name": "design.pes",
        },
        "operations": {
            "generate_low": True,
            "generate_high": False,
            "generate_video": False,
        },
        "output": {
            "bucket_name": "digitizer",
            "base_path": "processed/designs",
        },
    }


def test_render_from_space_size_guard(monkeypatch, client):
    payload = _base_payload()

    monkeypatch.setattr(render_service, "head_object_size", lambda *args, **kwargs: MAX_INPUT_BYTES + 1)

    def _unexpected_download(*args, **kwargs):
        raise AssertionError("download_to_file should not be invoked for oversized inputs")

    monkeypatch.setattr(render_service, "download_to_file", _unexpected_download)

    response = client.post("/render-from-space", json=payload)

    assert response.status_code == 413
    detail = response.json()["detail"]
    assert "job_id" in detail
    assert "exceeds the 50 MB limit" in detail["message"]


def test_render_from_space_only_fast(monkeypatch, tmp_path, client):
    payload = _base_payload()

    temp_processed = tmp_path / "processed"
    temp_processed.mkdir()
    monkeypatch.setattr(render_service, "processed_dir", str(temp_processed))

    sample_input = tmp_path / "design.pes"
    sample_input.write_bytes(b"PESDATA")

    monkeypatch.setattr(render_service, "head_object_size", lambda *args, **kwargs: 1024)

    def fake_download_to_file(bucket: str, key: str, dest: Path) -> None:
        dest.write_bytes(sample_input.read_bytes())

    def fake_run_blender_render(input_path: str, fast_output_path: str, **kwargs) -> float:
        Path(fast_output_path).write_bytes(b"FASTPNG")
        return 1.5

    def fake_upload_with_suffix_on_conflict(bucket: str, key: str, src_path: Path):
        src = Path(src_path)
        return {"final_key": key, "size_bytes": src.stat().st_size}

    monkeypatch.setattr(render_service, "download_to_file", fake_download_to_file)
    monkeypatch.setattr(render_service, "run_blender_render", fake_run_blender_render)
    monkeypatch.setattr(render_service, "upload_with_suffix_on_conflict", fake_upload_with_suffix_on_conflict)

    response = client.post("/render-from-space", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["errors"] == []
    assert "debug" not in body
    assert body["message"].endswith("File processing complete.")
    assert "Input file name: design.pes" in body["message"]
    assert 'Debug command: add "debug": true to the request payload to return detailed diagnostics.' in body["message"]
    outputs = body["outputs"]
    assert "fast_png" in outputs
    assert "legacy_png" not in outputs
    assert outputs["fast_png"]["object_key"].endswith("_fast.png")
    assert "Fast PNG return file location: processed/designs/design_fast.png" in body["message"]


def test_render_from_space_defaults_to_input_bucket_and_path(monkeypatch, tmp_path, client):
    payload = _base_payload()
    payload["output"] = {}

    temp_processed = tmp_path / "processed"
    temp_processed.mkdir()
    monkeypatch.setattr(render_service, "processed_dir", str(temp_processed))

    sample_input = tmp_path / "design.pes"
    sample_input.write_bytes(b"PESDATA")

    monkeypatch.setattr(render_service, "head_object_size", lambda *args, **kwargs: 1024)

    def fake_download_to_file(bucket: str, key: str, dest: Path) -> None:
        dest.write_bytes(sample_input.read_bytes())

    def fake_run_blender_render(input_path: str, fast_output_path: str, **kwargs) -> float:
        Path(fast_output_path).write_bytes(b"FASTPNG")
        return 1.0

    captured = {}

    def fake_upload_with_suffix_on_conflict(bucket: str, key: str, src_path: Path):
        src = Path(src_path)
        captured["bucket"] = bucket
        captured["key"] = key
        return {"final_key": key, "size_bytes": src.stat().st_size}

    monkeypatch.setattr(render_service, "download_to_file", fake_download_to_file)
    monkeypatch.setattr(render_service, "run_blender_render", fake_run_blender_render)
    monkeypatch.setattr(render_service, "upload_with_suffix_on_conflict", fake_upload_with_suffix_on_conflict)

    response = client.post("/render-from-space", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert captured["bucket"] == payload["input"]["bucket_name"]
    assert captured["key"] == "test/design_fast.png"
    assert body["outputs"]["bucket_name"] == payload["input"]["bucket_name"]
    assert body["outputs"]["base_path"] == payload["input"]["file_path"]
    assert "Input bucket: digitizer" in body["message"]
    assert "Fast PNG return bucket: digitizer" in body["message"]
    assert 'Debug command: add "debug": true to the request payload to return detailed diagnostics.' in body["message"]


def test_render_from_space_with_debug(monkeypatch, tmp_path, client):
    payload = _base_payload()
    payload["debug"] = True

    temp_processed = tmp_path / "processed"
    temp_processed.mkdir()
    monkeypatch.setattr(render_service, "processed_dir", str(temp_processed))

    sample_input = tmp_path / "design.pes"
    sample_input.write_bytes(b"PESDATA")

    monkeypatch.setattr(render_service, "head_object_size", lambda *args, **kwargs: 1024)
    monkeypatch.setattr(render_service, "_read_log_tail", lambda lines=60: "mock log tail")

    def fake_download_to_file(bucket: str, key: str, dest: Path) -> None:
        dest.write_bytes(sample_input.read_bytes())

    def fake_run_blender_render(input_path: str, fast_output_path: str, **kwargs) -> float:
        Path(fast_output_path).write_bytes(b"FASTPNG")
        return 2.0

    def fake_upload_with_suffix_on_conflict(bucket: str, key: str, src_path: Path):
        src = Path(src_path)
        return {"final_key": key, "size_bytes": src.stat().st_size}

    monkeypatch.setattr(render_service, "download_to_file", fake_download_to_file)
    monkeypatch.setattr(render_service, "run_blender_render", fake_run_blender_render)
    monkeypatch.setattr(render_service, "upload_with_suffix_on_conflict", fake_upload_with_suffix_on_conflict)

    response = client.post("/render-from-space", json=payload)
    assert response.status_code == 200

    body = response.json()
    assert "debug" in body
    debug_info = body["debug"]
    assert body["message"].endswith("File processing complete.")
    assert "Debug command executed: response includes debug diagnostics." in body["message"]
    assert debug_info["uploaded_assets"]["fast_png"]["object_key"].endswith("_fast.png")
    assert debug_info["stage_metrics"]["fast_png_seconds"] == 2.0
    assert debug_info["service_log_tail"] == "mock log tail"
