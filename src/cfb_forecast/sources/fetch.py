"""HTTP retrieval with robots.txt checks, per-host rate limits, retries and raw snapshots.

Every successful response is stored byte-for-byte under
``<data>/sources/raw/<source>/<UTC timestamp>.<ext>`` and appended to
``<data>/sources/raw/manifest.jsonl`` with its URL, status, SHA-256 and
retrieval time. Parsing always works from these saved files, so a collection
can be re-interpreted later (``sources-collect --offline``) without refetching.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .. import __version__
from ..logs import get

log = get("sources.fetch")

USER_AGENT = f"cfb-forecast/{__version__} (+https://github.com/bngomez98/cfb-forecast)"
BROWSER_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.8",
    "Accept-Encoding": "gzip",
}
EXTENSIONS = {"html": ".html", "json": ".json", "pdf": ".pdf", "csv": ".csv", "plain": ".txt", "xml": ".xml"}

_last_request = {}
_robots = {}


class FetchError(RuntimeError):
    """A source could not be retrieved; ``reason`` is a short machine-readable code."""

    def __init__(self, reason, message):
        super().__init__(message)
        self.reason = reason


@dataclass
class Snapshot:
    source: str
    url: str
    path: str
    retrieved_at: str
    status: int
    content_type: str
    sha256: str
    bytes: int

    def to_dict(self):
        return asdict(self)


def _wait(host, min_interval):
    last = _last_request.get(host)
    if last is not None:
        delay = min_interval - (time.monotonic() - last)
        if delay > 0:
            time.sleep(delay)
    _last_request[host] = time.monotonic()


def robots_allows(url, timeout=20):
    parts = urllib.parse.urlsplit(url)
    root = f"{parts.scheme}://{parts.netloc}"
    if root not in _robots:
        parser = urllib.robotparser.RobotFileParser()
        try:
            req = urllib.request.Request(f"{root}/robots.txt", headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                parser.parse(resp.read().decode("utf-8", errors="replace").splitlines())
        except urllib.error.HTTPError as exc:
            # RFC 9309: 4xx means no restrictions; 5xx means assume full disallow.
            parser.parse([] if 400 <= exc.code < 500 else ["User-agent: *", "Disallow: /"])
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise FetchError("unreachable", f"{root} unreachable: {exc}") from exc
        _robots[root] = parser
    return _robots[root].can_fetch(USER_AGENT, url)


def _content_kind(content_type, url):
    ct = (content_type or "").lower()
    for key in ("json", "pdf", "csv", "html", "xml", "plain"):
        if key in ct:
            return key
    suffix = Path(urllib.parse.urlsplit(url).path).suffix.lower().lstrip(".")
    return {"htm": "html", "txt": "plain"}.get(suffix, suffix or "html")


def fetch(source, url, data_root, min_interval=3.0, retries=3, timeout=30, respect_robots=True):
    """Download ``url`` politely and persist it. Raises FetchError on failure."""
    if respect_robots and not robots_allows(url):
        raise FetchError("robots-disallowed", f"robots.txt disallows {url} for {USER_AGENT}")
    host = urllib.parse.urlsplit(url).netloc
    headers = {"User-Agent": USER_AGENT, **BROWSER_HEADERS}
    last_error = None
    for attempt in range(retries):
        _wait(host, min_interval)
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
                if resp.headers.get("Content-Encoding", "").lower() == "gzip":
                    body = gzip.decompress(body)
                status, content_type = resp.status, resp.headers.get("Content-Type", "")
            break
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code in (401, 403, 451):
                raise FetchError(f"http-{exc.code}", f"{url} refused access (HTTP {exc.code}); the site blocks automated requests") from exc
            if exc.code == 404:
                raise FetchError("http-404", f"{url} not found (HTTP 404)") from exc
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            delay = float(retry_after) if retry_after and retry_after.isdigit() else 2 ** (attempt + 1)
            log.warning("%s: HTTP %s, retrying in %.0fs", source, exc.code, delay)
            time.sleep(delay)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            log.warning("%s: %s, retrying", source, exc)
            time.sleep(2 ** (attempt + 1))
    else:
        raise FetchError("unreachable", f"{url} failed after {retries} attempts: {last_error}")
    return store(source, url, body, content_type, data_root, status)


def store(source, url, body, content_type, data_root, status=200, retrieved_at=None):
    retrieved = retrieved_at or datetime.now(timezone.utc).replace(microsecond=0)
    kind = _content_kind(content_type, url)
    folder = Path(data_root) / "sources" / "raw" / source
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{retrieved:%Y%m%dT%H%M%SZ}{EXTENSIONS.get(kind, '.' + kind)}"
    path.write_bytes(body)
    snap = Snapshot(source, url, str(path), retrieved.isoformat(), int(status), content_type or kind,
                    hashlib.sha256(body).hexdigest(), len(body))
    with (Path(data_root) / "sources" / "raw" / "manifest.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(snap.to_dict()) + "\n")
    log.info("%s: saved %s (%d bytes, sha256 %s)", source, path, snap.bytes, snap.sha256[:12])
    return snap


def latest_snapshot(source, data_root):
    """Most recent stored snapshot for ``source`` (for offline re-interpretation)."""
    manifest = Path(data_root) / "sources" / "raw" / "manifest.jsonl"
    if not manifest.exists():
        return None
    latest = None
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entry = json.loads(line)
            if entry["source"] == source and Path(entry["path"]).exists():
                latest = entry
    return Snapshot(**latest) if latest else None
