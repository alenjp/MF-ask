"""
Stage 1 of ingestion: LOAD.

Fetches the 5 public HDFC scheme pages listed in sources.csv, strips boilerplate
(nav / footer / SEO link dumps), and writes one cleaned .txt per page to data/raw/.

Public pages only. No login, no app back-end, no PII.
"""

from __future__ import annotations

import csv
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
SOURCES_CSV = ROOT / "sources.csv"
RAW_DIR = ROOT / "data" / "raw"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-IN,en;q=0.9",
}

# Boilerplate that carries no scheme facts. Removed before chunking.
DROP_SELECTORS = [
    "header",
    "footer",
    "nav",
    "noscript",
    "script",
    "style",
    "aside",
    "form",
    "svg",
    "[class*='Footer']",
    "[class*='Navbar']",
    "[class*='Header']",
    "[class*='Cookie']",
]

# Section markers used to slice the page body out of its nav/footer shell.
FOOTER_MARKS = ("About Us", "Version:", "© 2016")

STOP_BOILERPLATE_PREFIXES = (
    "Stocks Intraday ETF Screener Show more F&O Show More MTFs Margin calculator "
    "Watchlist IPO F&O Margin MTF MTF calculator IPO Apply Mutual Funds AMCs "
    "ETFs Corporate Bonds IPO ETFs Debt Mutual Funds Dividend Yield Mutual Funds "
    "Small Cap Funds Index Funds Hybrid Funds Corporate Bonds ETFs Direct Equity"
)


def _clean_lines(text: str) -> list[str]:
    out: list[str] = []
    for raw_line in text.splitlines():
        line = re.sub(r"[ \t\u00a0]+", " ", raw_line).strip()
        if not line:
            if out and out[-1] != "":
                out.append("")
            continue
        if len(line) <= 2 and not line.isdigit():
            continue
        out.append(line)
    while out and out[-1] == "":
        out.pop()
    return out


def _slice_to_fund_facts(lines: list[str]) -> list[str]:
    """Keep the page body, drop the mega nav above and the footer link dump below."""
    start = 0
    end = len(lines)
    for i in range(start, len(lines)):
        if any(lines[i].startswith(mark) for mark in FOOTER_MARKS):
            end = i
            break
    body = lines[start:end]
    while body and not body[0]:
        body.pop(0)
    return body


def _is_nav_noise(line: str) -> bool:
    """True for the pure slash-separated mega-menu link rows that survive nav stripping."""
    return line.count("/") >= 3 and len(line) > 60


def _drop_boilerplate_runs(lines: list[str]) -> list[str]:
    """Remove the mega-menu link rows that the nav stripper leaves behind."""
    return [line for line in lines if not _is_nav_noise(line)]


def extract(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    if not resp.encoding or resp.encoding.lower() == "iso-8859-1":
        resp.encoding = resp.apparent_encoding or "utf-8"
    soup = BeautifulSoup(resp.text, "lxml")

    for sel in DROP_SELECTORS:
        for node in soup.select(sel):
            node.decompose()

    text = soup.get_text("\n")
    lines = _clean_lines(text)
    lines = _slice_to_fund_facts(lines)
    lines = _drop_boilerplate_runs(lines)
    return "\n".join(lines)


def slug(url: str) -> str:
    return url.rstrip("/").rsplit("/", 1)[-1]


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(SOURCES_CSV.open(encoding="utf-8")))
    if len(rows) != 5:
        print(f"[load] expected 5 sources, found {len(rows)}", file=sys.stderr)
        return 1

    for row in rows:
        url, name = row["url"], row["scheme"]
        target = RAW_DIR / f"{slug(url)}.txt"
        if target.exists() and target.stat().st_size > 1500:
            print(f"[load] cached  {name}: {target.name} ({target.stat().st_size} bytes)")
            continue
        try:
            body = extract(url)
        except Exception as exc:  # noqa: BLE001
            print(f"[load] FAILED  {name}: {exc}", file=sys.stderr)
            return 2
        header = (
            f"SOURCE_URL: {url}\n"
            f"SCHEME: {row['scheme']}\n"
            f"CATEGORY: {row['category']}\n"
            f"PLAN: {row['plan']}\n"
            f"{'-' * 72}\n\n"
        )
        target.write_text(header + body, encoding="utf-8")
        print(f"[load] saved   {name}: {target.name} ({len(body)} chars)")
        time.sleep(1.5)

    print(f"[load] done -> {RAW_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
