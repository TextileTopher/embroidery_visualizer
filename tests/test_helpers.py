from unittest.mock import MagicMock

import pytest

import render_service
from render_service import (
    SpacesCollisionError,
    compose_key,
    upload_with_suffix_on_conflict,
)


def test_compose_key_with_path():
    assert compose_key("folder/sub", "design.pes") == "folder/sub/design.pes"


def test_compose_key_without_path():
    assert compose_key(None, "design.pes") == "design.pes"


def test_upload_with_suffix_on_conflict_appends_timestamp(monkeypatch, tmp_path):
    src = tmp_path / "sample.png"
    src.write_bytes(b"binarydata")

    uploaded = {}

    class DummyClient:
        def upload_file(self, src_path: str, bucket: str, key: str, *, ExtraArgs=None):
            uploaded["bucket"] = bucket
            uploaded["key"] = key
            uploaded["src_path"] = src_path
            uploaded["extra_args"] = ExtraArgs

    dummy_client = DummyClient()
    monkeypatch.setattr(render_service, "get_s3_client", lambda: dummy_client)

    exists_states = iter([True, False])

    def fake_key_exists(bucket: str, key: str) -> bool:
        return next(exists_states)

    monkeypatch.setattr(render_service, "_key_exists", fake_key_exists)

    result = upload_with_suffix_on_conflict("digitizer", "processed/xmass_fast.png", src)

    assert uploaded["bucket"] == "digitizer"
    assert uploaded["src_path"] == str(src)
    assert uploaded["key"] != "processed/xmass_fast.png"
    assert uploaded["key"].startswith("processed/xmass_fast")
    assert uploaded["extra_args"] == {"ACL": "public-read"}
    assert result["final_key"] == uploaded["key"]
    assert result["size_bytes"] == src.stat().st_size


def test_upload_with_suffix_on_conflict_unresolved(monkeypatch, tmp_path):
    src = tmp_path / "sample.png"
    src.write_bytes(b"binarydata")

    monkeypatch.setattr(render_service, "get_s3_client", lambda: MagicMock())
    monkeypatch.setattr(render_service, "_key_exists", lambda bucket, key: True)

    with pytest.raises(SpacesCollisionError):
        upload_with_suffix_on_conflict("digitizer", "processed/xmass_fast.png", src)
