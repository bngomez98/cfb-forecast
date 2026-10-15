"""Turn a raw page (HTML, JSON, plain text or PDF) into candidate text lines.

The interpreter works on lines, so every format is reduced to the same shape:

* HTML tables become one line per row with cells joined by " | ". Cell
  ``data-stat`` attributes (Sports-Reference style) are kept in ``rows``.
* Other visible HTML text becomes one line per block element.
* ``application/ld+json`` (schema.org SportsEvent), ``__NEXT_DATA__`` and other
  embedded JSON scripts are parsed. Objects with home/away structure become
  lines where every number sits next to its own team.
* PDFs are converted to text with ``pypdf``.

Only the standard library is needed for HTML.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

BLOCK = {"p", "div", "li", "tr", "br", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article",
         "header", "footer", "table", "thead", "tbody", "ul", "ol", "dd", "dt"}
SKIP = {"style", "noscript", "svg", "template", "head"}
JSON_SCRIPT_TYPES = {"application/ld+json", "application/json"}


@dataclass
class Document:
    lines: list[str] = field(default_factory=list)
    rows: list[list[dict]] = field(default_factory=list)  # table rows: [{"text":..., "stat":...}]
    json_blobs: list = field(default_factory=list)
    title: str = ""


class _Parser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.doc = Document()
        self._buf = []
        self._skip = 0
        self._script = None
        self._script_buf = []
        self._cell = None
        self._row = None
        self._in_title = False

    def _flush(self):
        text = re.sub(r"\s+", " ", "".join(self._buf)).strip()
        if text:
            self.doc.lines.append(text)
        self._buf = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "script":
            kind = (attrs.get("type") or "").lower()
            self._script = "json" if kind in JSON_SCRIPT_TYPES or attrs.get("id") == "__NEXT_DATA__" else "js"
            self._script_buf = []
            return
        if tag in SKIP:
            self._skip += 1
            return
        if tag == "title":
            self._in_title = True
        if tag == "tr":
            self._flush()
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = {"text": "", "stat": attrs.get("data-stat")}
        elif tag in BLOCK and self._row is None:
            self._flush()

    def handle_endtag(self, tag):
        if tag == "script":
            if self._script == "json":
                raw = "".join(self._script_buf).strip()
                try:
                    self.doc.json_blobs.append(json.loads(raw))
                except ValueError:
                    pass
            self._script = None
            return
        if tag in SKIP:
            self._skip = max(0, self._skip - 1)
            return
        if tag == "title":
            self._in_title = False
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._cell["text"] = re.sub(r"\s+", " ", self._cell["text"]).strip()
            self._row.append(self._cell)
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if any(c["text"] for c in self._row):
                self.doc.rows.append(self._row)
                self.doc.lines.append(" | ".join(c["text"] for c in self._row))
            self._row = None
            self._buf = []
        elif tag in BLOCK and self._row is None:
            self._flush()

    def handle_data(self, data):
        if self._script is not None:
            self._script_buf.append(data)
            return
        if self._skip:
            return
        if self._in_title:
            self.doc.title += data
        if self._cell is not None:
            self._cell["text"] += data
        elif self._row is None:
            self._buf.append(data)

    def close(self):
        super().close()
        self._flush()
        return self.doc


_SCORE_KEY = re.compile(r"(?:^|_)(?:score|points|pts|final)(?:$|_)|^score$|Score$|Points$", re.I)
_NAME_KEY = re.compile(r"name|team|school|display|location|abbrev", re.I)
_PROB_KEY = re.compile(r"prob|win_?pct|chance|likelihood", re.I)
_SPREAD_KEY = re.compile(r"spread|line(?!up)|handicap", re.I)
_FINAL_KEY = re.compile(r"status|state|completed|final", re.I)


def _walk(obj, depth=0, limit=4):
    """Yield (key, value) for scalar leaves up to ``limit`` levels deep."""
    if depth > limit:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                yield from _walk(v, depth + 1, limit)
            else:
                yield str(k), v
    elif isinstance(obj, list):
        for v in obj[:6]:
            yield from _walk(v, depth + 1, limit)


def _side(sub):
    """Summarize one team's subtree: candidate names, score, probability, spread."""
    names, score, prob, spread = [], None, None, None
    for k, v in _walk(sub):
        if isinstance(v, bool) or v is None:
            continue
        if isinstance(v, str) and _NAME_KEY.search(k) and not re.search(r"id$|url|logo|color|image|href", k, re.I):
            names.append(v.strip())
        num = v if isinstance(v, (int, float)) else (float(v) if isinstance(v, str) and re.fullmatch(r"[+-]?\d+(?:\.\d+)?", v.strip()) else None)
        if num is None:
            continue
        if score is None and _SCORE_KEY.search(k) and 0 <= num < 100:
            score = num
        elif prob is None and _PROB_KEY.search(k) and 0 <= num <= 100:
            prob = num * 100 if num <= 1 else num
        elif spread is None and _SPREAD_KEY.search(k) and abs(num) <= 60:
            spread = num
    return names, score, prob, spread


def _fmt(names, score, prob, spread):
    parts = [max(names, key=len) if names else ""]
    if spread is not None:
        parts.append(f"{spread:+g}")
    if score is not None:
        parts.append(f"{score:g}")
    if prob is not None:
        parts.append(f"{prob:.1f}%")
    return " ".join(parts)


def semantic_json_lines(blob, max_nodes=50000):
    """Game lines built from home/away structure, so each number stays with its own team.

    Recognizes objects with ``home*``/``away*`` keys (``homeTeam``, ``home_score``...)
    and ESPN-style ``competitors`` lists whose items carry ``homeAway``.
    Each emitted line is ``"<away> [spread] [score] [prob%] | <home> ..."`` plus
    ``final`` when a status field says so.
    """
    lines, stack, seen = [], [blob], 0
    while stack and seen < max_nodes:
        node = stack.pop()
        seen += 1
        if isinstance(node, list):
            stack.extend(v for v in node if isinstance(v, (dict, list)))
            continue
        if not isinstance(node, dict):
            continue
        home, away = {}, {}
        for k, v in node.items():
            key = k.lower()
            if key.startswith("home") or key.endswith("home") or "_home" in key:
                home[k] = v
            elif key.startswith("away") or key.endswith("away") or "_away" in key or key.startswith("visitor"):
                away[k] = v
        comps = node.get("competitors")
        if isinstance(comps, list):
            for c in comps:
                if isinstance(c, dict):
                    side = str(c.get("homeAway") or c.get("home_away") or ("home" if c.get("isHome") else "")).lower()
                    if side == "home":
                        home["c"] = c
                    elif side == "away":
                        away["c"] = c
        if home and away:
            h, a = _side(home), _side(away)
            if h[0] and a[0]:
                status = " ".join(str(v) for k, v in _walk({k: v for k, v in node.items() if k not in home and k not in away}, limit=3) if _FINAL_KEY.search(k))
                final = bool(re.search(r"\bfinal\b|\bpost\b|\btrue\b|completed", status, re.I))
                lines.append(f"{_fmt(*a)} | {_fmt(*h)}" + (" | final" if final else ""))
        stack.extend(v for v in node.values() if isinstance(v, (dict, list)))
    return lines


def sports_events(blob):
    """schema.org SportsEvent objects from JSON-LD, as (away, home, start, away_score, home_score) lines."""
    out, stack = [], [blob]
    while stack:
        node = stack.pop()
        if isinstance(node, list):
            stack.extend(node)
        elif isinstance(node, dict):
            kinds = node.get("@type")
            kinds = kinds if isinstance(kinds, list) else [kinds]
            if "SportsEvent" in kinds and isinstance(node.get("homeTeam"), (dict, str)) and isinstance(node.get("awayTeam"), (dict, str)):
                def name(t):
                    return t.get("name", "") if isinstance(t, dict) else str(t)
                out.append(f"{name(node['awayTeam'])} at {name(node['homeTeam'])} | {node.get('startDate', '')}")
            stack.extend(v for v in node.values() if isinstance(v, (dict, list)))
    return out


def parse_html(text):
    parser = _Parser()
    parser.feed(text)
    doc = parser.close()
    for blob in doc.json_blobs:
        doc.lines.extend(sports_events(blob))
        doc.lines.extend(semantic_json_lines(blob))
    return doc


def parse_json(text):
    blob = json.loads(text)
    return Document(lines=sports_events(blob) + semantic_json_lines(blob), json_blobs=[blob])


def parse_text(text):
    return Document(lines=[re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()])


def parse_pdf(path):
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError("PDF sources need pypdf (a core dependency): pip install pypdf") from exc
    reader = PdfReader(str(path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return parse_text(text)


def parse_file(path, content_type=None):
    path = Path(path)
    suffix = path.suffix.lower()
    kind = (content_type or "").lower()
    if suffix == ".pdf" or "pdf" in kind:
        return parse_pdf(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".json" or "json" in kind:
        try:
            return parse_json(text)
        except ValueError:
            pass
    if suffix in {".html", ".htm"} or "html" in kind or text.lstrip()[:1] == "<":
        return parse_html(text)
    return parse_text(text)
