"""Source receipts bind forecasts to actual stored bytes, never claimed history."""
import gzip
import io
import json
from pathlib import Path
import urllib.error

import pandas as pd
import pytest

from cfb_forecast.ingest import data


def schedule():
    return pd.DataFrame([dict(game_id=1, season=2026, week=1,
        start_date="2026-09-05T16:00:00Z", completed=False, neutral_site=False,
        home_id=1, away_id=2, home_team="A", away_team="B",
        home_points=None, away_points=None, fbs_game=True)])


def pin(tmp_path, filename, content, url):
    import hashlib
    receipt = dict(url=url, sha256=hashlib.sha256(content).hexdigest(), bytes=len(content),
                   downloaded_at=None, provenance="Retained snapshot")
    manifest = tmp_path / "pin.json"
    manifest.write_text(json.dumps({"files": {filename: receipt}}))
    return manifest


@pytest.mark.parametrize("compressed", [False, True])
def test_fetch_pinned_source_checks_stored_bytes_and_preserves_provenance(tmp_path, monkeypatch, compressed):
    content = schedule().to_csv(index=False).encode()
    url = "https://example.org/schedule.csv" + (".gz" if compressed else "")
    manifest = pin(tmp_path, "schedule_2026.csv", content, url)
    root = tmp_path / "data"
    root.mkdir()
    # No advanced-stat download needed for this test.
    (root / "adv_team_2026.parquet").write_bytes(b"cached")
    obj = json.loads(manifest.read_text())
    import hashlib
    obj["files"]["adv_team_2026.parquet"] = dict(url="https://example.org/adv", sha256=hashlib.sha256(b"cached").hexdigest(), bytes=6)
    manifest.write_text(json.dumps(obj))
    requests = []
    def download(request, timeout):
        requests.append(request.full_url)
        return io.BytesIO(gzip.compress(content) if compressed else content)
    monkeypatch.setattr(data.urllib.request, "urlopen", download)
    result = data.fetch(root, [2026], manifest_path=manifest)
    assert requests == [url]
    assert (root / "schedule_2026.csv").read_bytes() == content
    assert result["files"]["schedule_2026.csv"]["provenance"] == "Retained snapshot"
    assert result["files"]["schedule_2026.csv"]["snapshot_downloaded_at"] is None


def test_cached_file_tampering_is_rejected_without_network(tmp_path, monkeypatch):
    (tmp_path / "schedule_2026.csv").write_bytes(b"modified")
    manifest = pin(tmp_path, "schedule_2026.csv", b"original", "https://example.org/data")
    (tmp_path / "source_manifest.json").write_text(manifest.read_text())
    def forbidden(*args, **kwargs):
        pytest.fail("A corrupt cache must fail before network access")
    monkeypatch.setattr(data.urllib.request, "urlopen", forbidden)
    with pytest.raises(ValueError, match="checksum mismatch"):
        data.fetch(tmp_path, [2026])
    with pytest.raises(ValueError, match="checksum mismatch"):
        data.load_data(tmp_path)


def test_failed_pinned_refresh_preserves_previous_file_and_receipt(tmp_path, monkeypatch):
    original = b"previous snapshot"
    target = tmp_path / "schedule_2026.csv"
    target.write_bytes(original)
    saved = '{"files": {}}'
    (tmp_path / "source_manifest.json").write_text(saved)
    manifest = pin(tmp_path, target.name, b"wrong hash", "https://example.org/schedule.csv")
    monkeypatch.setattr(data.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(schedule().to_csv(index=False).encode()))
    monkeypatch.setattr(data.time, "sleep", lambda _: None)
    with pytest.raises(ValueError, match="mismatch"):
        data.fetch(tmp_path, [2026], refresh=True, manifest_path=manifest)
    assert target.read_bytes() == original
    assert (tmp_path / "source_manifest.json").read_text() == saved
    assert not target.with_suffix(".download").exists()


def test_pinned_missing_source_fails_instead_of_recording_unavailable(tmp_path, monkeypatch):
    manifest = pin(tmp_path, "schedule_2026.csv", b"snapshot", "https://example.org/missing")
    def unavailable(*a, **kw):
        raise urllib.error.HTTPError("https://example.org/missing", 404, "Missing", {}, None)
    monkeypatch.setattr(data.urllib.request, "urlopen", unavailable)
    monkeypatch.setattr(data.time, "sleep", lambda _: None)
    with pytest.raises(urllib.error.HTTPError):
        data.fetch(tmp_path / "data", [2026], manifest_path=manifest)


@pytest.mark.parametrize("receipt", [{"sha256": "wrong"}, {"bytes": -1}, {"bytes": True}])
def test_malformed_receipts_are_rejected(tmp_path, receipt):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"files": {"schedule.csv": receipt}}))
    with pytest.raises(ValueError):
        data.read_source_manifest(path)
