"""Tests for the Modal cloud-GPU backend wiring.

These cover the local-side orchestration logic — zip round-trip, availability
gating, and the error path — without requiring the `modal` SDK or any network.
"""

from __future__ import annotations

import pytest

from workers import modal_runner


def test_zip_roundtrip(tmp_path):
    src = tmp_path / "adapter"
    src.mkdir()
    (src / "adapter_config.json").write_text('{"r": 16}')
    (src / "weights.bin").write_bytes(b"\x00\x01\x02")
    (src / "nested").mkdir()
    (src / "nested" / "extra.txt").write_text("hello")

    blob = modal_runner._zip_dir(str(src))
    assert isinstance(blob, bytes) and blob

    dest = tmp_path / "restored"
    modal_runner._unzip_to(blob, str(dest))
    assert (dest / "adapter_config.json").read_text() == '{"r": 16}'
    assert (dest / "weights.bin").read_bytes() == b"\x00\x01\x02"
    assert (dest / "nested" / "extra.txt").read_text() == "hello"


def test_modal_unavailable_without_credentials(monkeypatch):
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    assert modal_runner.modal_available() is False


def test_run_on_modal_raises_when_unavailable(monkeypatch, tmp_path):
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    with pytest.raises(RuntimeError, match="Modal backend"):
        modal_runner.run_on_modal(
            job_id="j1",
            model_cfg={},
            lora_cfg={},
            train_cfg={"output_dir": str(tmp_path)},
            dataset_path="",
            output_path=str(tmp_path / "out"),
        )


def test_modal_available_requires_sdk_and_tokens(monkeypatch):
    monkeypatch.setenv("MODAL_TOKEN_ID", "id")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "secret")
    # Mirrors real availability: only True when the SDK actually imported.
    assert modal_runner.modal_available() is (modal_runner.modal is not None)
