"""
download_filings.py — Download sample 10-K filings from SEC EDGAR.

Three access modes:
  1. EDGAR Full-Text Search API (real-time, no key needed)
  2. EDGAR Company Facts API (XBRL data only, instant access)
  3. sec-edgar-downloader (bulk local download of raw filings)

Live streaming:
  The EDGAR EFTS (Full-Text Search) RSS feed is polled for new 10-K
  filings. Use --watch to stream new filings as they are filed.

Usage:
  python download_filings.py --sample          # Download 20 sample filings
  python download_filings.py --watch           # Stream new 10-K filings live
  python download_filings.py --cik 0000320193  # Download all 10-Ks for one company
  python download_filings.py --ticker AAPL     # Download by ticker
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Iterator, Optional

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# EDGAR API constants (all public, no API key required)
# ---------------------------------------------------------------------------

# Rate limit: SEC requires <= 10 requests/second; we use 0.12s delay = ~8 req/s
EDGAR_RATE_DELAY = 0.12

EDGAR_BASE         = "https://data.sec.gov"
EDGAR_SUBMISSIONS  = f"{EDGAR_BASE}/submissions/CIK{{cik}}.json"
EDGAR_COMPANY_FACTS= f"{EDGAR_BASE}/api/xbrl/companyfacts/CIK{{cik}}.json"
EDGAR_COMPANY_CONCEPT = f"{EDGAR_BASE}/api/xbrl/companyconcept/CIK{{cik}}/{{taxonomy}}/{{tag}}.json"
EDGAR_EFTS_SEARCH  = "https://efts.sec.gov/LATEST/search-index?q=%2210-K%22&dateRange=custom&startdt={start}&enddt={end}&forms=10-K"
EDGAR_EFTS_RSS     = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=10-K&dateb=&owner=include&count=40&search_text=&output=atom"
EDGAR_ARCHIVES     = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/"

# User-Agent is required by SEC (use your real email)
DEFAULT_HEADERS = {
    "User-Agent": "edgar-rl-research research@example.com",
    "Accept-Encoding": "gzip, deflate",
    # Note: do NOT set Host here — it applies to all domains and breaks www.sec.gov requests
}

FILINGS_DIR = Path(__file__).parent / "filings"

# ---------------------------------------------------------------------------
# Sample corpus: mix of large-cap and mid-cap across sectors
# ---------------------------------------------------------------------------

SAMPLE_TICKERS = [
    # Tech (large-cap)
    ("0000320193", "AAPL",  "Apple Inc"),
    ("0000789019", "MSFT",  "Microsoft Corp"),
    ("0001652044", "GOOGL", "Alphabet Inc"),
    ("0001018724", "AMZN",  "Amazon.com Inc"),
    ("0001326801", "META",  "Meta Platforms Inc"),
    # Tech (mid-cap / smaller)
    ("0001045810", "NVDA",  "NVIDIA Corp"),
    ("0000796343", "ADBE",  "Adobe Inc"),
    ("0001640147", "SNOW",  "Snowflake Inc"),
    # Industrials
    ("0000040545", "GE",    "General Electric Co"),
    ("0000018230", "CAT",   "Caterpillar Inc"),
    ("0000066740", "MMM",   "3M Co"),
    # Financials
    ("0000070858", "BAC",   "Bank of America Corp"),
    ("0000019617", "JPM",   "JPMorgan Chase & Co"),
    ("0000051143", "IBM",   "IBM Corp"),
    # Healthcare
    ("0000200406", "JNJ",   "Johnson & Johnson"),
    ("0000078003", "PFE",   "Pfizer Inc"),
    # Consumer
    ("0000021344", "KO",    "Coca-Cola Co"),
    ("0000080424", "PG",    "Procter & Gamble Co"),
    # Energy
    ("0000034088", "XOM",   "Exxon Mobil Corp"),
    ("0000093410", "CVX",   "Chevron Corp"),
]


# ---------------------------------------------------------------------------
# HTTP session with SEC rate limiting
# ---------------------------------------------------------------------------

class EdgarSession:
    def __init__(self, user_agent: str = DEFAULT_HEADERS["User-Agent"]):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
        })
        self._last_request = 0.0

    def get(self, url: str, **kwargs) -> requests.Response:
        elapsed = time.time() - self._last_request
        if elapsed < EDGAR_RATE_DELAY:
            time.sleep(EDGAR_RATE_DELAY - elapsed)
        resp = self.session.get(url, timeout=30, **kwargs)
        self._last_request = time.time()
        resp.raise_for_status()
        return resp

    def get_json(self, url: str, **kwargs) -> dict:
        # data.sec.gov requires explicit Host header; www.sec.gov does not
        if "data.sec.gov" in url and "headers" not in kwargs:
            kwargs["headers"] = {"Host": "data.sec.gov"}
        resp = self.get(url, **kwargs)
        return resp.json()


# ---------------------------------------------------------------------------
# CIK ↔ Ticker resolution
# ---------------------------------------------------------------------------

def resolve_ticker_to_cik(ticker: str, session: EdgarSession) -> Optional[str]:
    """Use EDGAR company search to resolve a ticker to a zero-padded CIK."""
    url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&forms=10-K"
    try:
        resp = session.get(
            f"https://www.sec.gov/cgi-bin/browse-edgar?company={ticker}&CIK=&type=10-K&dateb=&owner=include&count=10&search_text=&action=getcompany&output=atom"
        )
        # Also works: https://data.sec.gov/submissions/CIK{padded}.json
        # Fallback: use the company tickers JSON file
        tickers_resp = session.get_json("https://www.sec.gov/files/company_tickers.json")
        for entry in tickers_resp.values():
            if entry.get("ticker", "").upper() == ticker.upper():
                return str(entry["cik_str"]).zfill(10)
    except Exception as e:
        logger.warning("Could not resolve ticker %s: %s", ticker, e)
    return None


# ---------------------------------------------------------------------------
# Submission metadata
# ---------------------------------------------------------------------------

def get_submissions(cik: str, session: EdgarSession) -> dict:
    """Fetch filing submission history for a CIK."""
    padded = cik.zfill(10)
    url = EDGAR_SUBMISSIONS.format(cik=padded)
    return session.get_json(url, headers={"Host": "data.sec.gov"})


def get_10k_accessions(
    submissions: dict,
    max_filings: int = 3,
) -> list[dict]:
    """Extract up to max_filings 10-K accession numbers from a submissions record."""
    recent = submissions.get("filings", {}).get("recent", {})
    forms    = recent.get("form", [])
    accns    = recent.get("accessionNumber", [])
    dates    = recent.get("filingDate", [])
    reports  = recent.get("reportDate", [])

    results = []
    for form, accn, date, report in zip(forms, accns, dates, reports):
        if form == "10-K" and accn:
            results.append({
                "accession": accn,
                "filed_date": date,
                "report_date": report,
            })
        if len(results) >= max_filings:
            break
    return results


# ---------------------------------------------------------------------------
# Filing index & document discovery
# ---------------------------------------------------------------------------

def get_filing_index(cik: str, accession: str, session: EdgarSession) -> dict:
    """Fetch the filing index JSON for a specific accession."""
    padded_cik = cik.zfill(10)
    accn_nodash = accession.replace("-", "")
    url = f"{EDGAR_BASE}/submissions/{padded_cik}/{accn_nodash}.json"
    # Alternative: use the filing-index page
    try:
        return session.get_json(url)
    except Exception:
        pass

    # Fallback: parse the index page
    index_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accn_nodash}/{accession}-index.json"
    try:
        return session.get_json(index_url)
    except Exception:
        return {}


def find_filing_documents(
    cik: str,
    accession: str,
    session: EdgarSession,
) -> dict[str, Optional[str]]:
    """
    Return paths to {xbrl, html, txt} documents for a filing.
    Uses the EDGAR filing index.
    """
    padded_cik = cik.zfill(10)
    accn_nodash = accession.replace("-", "")
    base_url = EDGAR_ARCHIVES.format(cik=int(cik), accession_nodash=accn_nodash)

    # Parse the EDGAR directory listing HTML to discover filing documents
    try:
        from bs4 import BeautifulSoup
        resp = session.get(base_url)
        soup = BeautifulSoup(resp.text, "lxml")
    except Exception as e:
        logger.warning("Could not fetch directory for %s / %s: %s", cik, accession, e)
        return {"xbrl": None, "html": None, "txt": None}

    # Collect only filenames that live inside this accession directory
    # (filter out SEC navigation hrefs which are relative paths like /about.shtml)
    all_files: list[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Only keep hrefs that are plain filenames (no slashes or absolute paths)
        fname = href.split("/")[-1]
        if fname and "." in fname and not href.startswith("/") and not href.startswith("http"):
            all_files.append(fname)
        # Also accept absolute paths that are inside this accession directory
        elif href.startswith("/Archives/edgar/data/") and accn_nodash in href:
            all_files.append(href.split("/")[-1])

    xbrl_url = html_url = txt_url = None

    # XBRL instance document priority:
    #   1. *_htm.xml  — inline XBRL instance (most common post-2018)
    #   2. *.xml      — traditional XBRL instance (not a linkbase)
    _linkbase_suffixes = ("_cal.xml", "_def.xml", "_lab.xml", "_pre.xml", ".xsd")
    _skip_prefixes = ("R", "FilingSummary", "MetaLinks", "report", "Show")

    for fname in all_files:
        if fname.endswith("_htm.xml") and xbrl_url is None:
            xbrl_url = base_url + fname
        elif (fname.endswith(".xml")
              and not any(fname.endswith(s) for s in _linkbase_suffixes)
              and not any(fname.startswith(p) for p in _skip_prefixes)
              and xbrl_url is None):
            xbrl_url = base_url + fname

    # Main 10-K HTML: prefer {ticker}-{date}.htm (standard EDGAR naming for primary document)
    # Exclude R*.htm viewer fragments and exhibit/consent files
    _exhibit_patterns = ("exhibit", "ex-", "ex1", "ex2", "ex3", "ex4", "ex9",
                         "consent", "auditor", "certif", "power")
    _primary_re = re.compile(r"^[a-z]+-\d{8}\.htm$", re.IGNORECASE)

    # Pass 1: prefer canonical {ticker}-{date}.htm pattern
    for fname in all_files:
        if _primary_re.match(fname) and html_url is None:
            html_url = base_url + fname

    # Pass 2: fallback to any non-exhibit, non-fragment .htm
    if html_url is None:
        for fname in all_files:
            if not fname.endswith(".htm"):
                continue
            if fname.startswith("R") and fname[1:].split(".")[0].isdigit():
                continue
            if any(p in fname.lower() for p in _exhibit_patterns):
                continue
            if fname.endswith("-index.htm") or fname.endswith("-index-headers.html"):
                continue
            html_url = base_url + fname
            break

    # Full submission TXT: {accession}.txt
    accn_txt = f"{accession}.txt"
    if accn_txt in all_files:
        txt_url = base_url + accn_txt
    else:
        for fname in all_files:
            if fname.endswith(".txt") and txt_url is None:
                txt_url = base_url + fname

    return {"xbrl": xbrl_url, "html": html_url, "txt": txt_url}


# ---------------------------------------------------------------------------
# 10-K verification — parse SGML header from the full-submission TXT
# ---------------------------------------------------------------------------

def parse_sgml_header(content: bytes) -> dict[str, str]:
    """
    Parse the EDGAR SGML submission header that appears at the very top of
    every full-submission .txt file before the actual documents begin.

    Structure (from SEC spec):
        <SEC-DOCUMENT>...
        <SEC-HEADER>...
        CONFORMED SUBMISSION TYPE:    10-K
        FORM TYPE:                    10-K
        CONFORMED PERIOD OF REPORT:   20251231
        ...
        </SEC-HEADER>

    Returns a flat dict of every "KEY:  value" pair found in the header.
    Stops parsing at </SEC-HEADER> or <DOCUMENT> to avoid reading the body.
    """
    # Read just the header — it's always in the first ~8 KB
    head_bytes = content[:8192]
    head_text = head_bytes.decode("utf-8", errors="replace")

    fields: dict[str, str] = {}
    in_header = False

    for line in head_text.splitlines():
        stripped = line.strip()

        if "<SEC-HEADER>" in stripped or "<SEC-DOCUMENT>" in stripped:
            in_header = True
            continue
        if "</SEC-HEADER>" in stripped or "<DOCUMENT>" in stripped:
            break

        if in_header and ":" in stripped:
            # Lines look like:  "CONFORMED SUBMISSION TYPE:    10-K"
            key, _, value = stripped.partition(":")
            key = key.strip().upper()
            value = value.strip()
            if key and value:
                fields[key] = value

    return fields


def is_10k_submission(txt_content: bytes) -> bool:
    """
    Return True only if the SGML header's CONFORMED SUBMISSION TYPE is 10-K.
    Accepts 10-K and 10-K/A (amended). Rejects 10-KSB, 20-F, DEF 14A, etc.
    Also falls back to checking FORM TYPE if CONFORMED SUBMISSION TYPE is absent.
    Returns True (don't reject) if the header is entirely missing.
    """
    header = parse_sgml_header(txt_content)

    submission_type = (
        header.get("CONFORMED SUBMISSION TYPE")
        or header.get("FORM TYPE")
        or ""
    )

    if not submission_type:
        # No header found — don't reject (e.g. unit tests, mock data)
        return True

    # Accept 10-K and 10-K/A only
    return submission_type in ("10-K", "10-K/A")


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------

def download_filing(
    cik: str,
    accession: str,
    out_dir: Path,
    session: EdgarSession,
) -> dict[str, Optional[Path]]:
    """
    Download XBRL, HTML, and TXT documents for one filing.
    Verifies the TXT submission header confirms form type = 10-K before saving.
    Returns {xbrl: Path, html: Path, txt: Path} (None if not found/downloaded).
    """
    docs = find_filing_documents(cik, accession, session)
    saved: dict[str, Optional[Path]] = {}

    # --- Download and verify the TXT first (cheapest header check) ---
    txt_url = docs.get("txt")
    txt_path: Optional[Path] = None

    if txt_url:
        filename = txt_url.split("/")[-1]
        out_path = out_dir / filename
        if out_path.exists():
            logger.info("  [cached] %s", filename)
            txt_content = out_path.read_bytes()
        else:
            try:
                resp = session.get(txt_url)
                txt_content = resp.content
            except Exception as e:
                logger.warning("  [fail  ] %s — %s", filename, e)
                txt_content = b""

        if txt_content:
            if not is_10k_submission(txt_content):
                header = parse_sgml_header(txt_content)
                form = header.get("CONFORMED SUBMISSION TYPE", "unknown")
                logger.warning(
                    "  [skip  ] %s is form type '%s', not 10-K — skipping filing",
                    filename, form,
                )
                return {"xbrl": None, "html": None, "txt": None}

            if not out_path.exists():
                out_path.write_bytes(txt_content)
                logger.info("  [saved ] %s (%d KB)", filename, len(txt_content) // 1024)
            txt_path = out_path

    saved["txt"] = txt_path

    # --- Download remaining documents ---
    for doc_type in ("xbrl", "html"):
        url = docs.get(doc_type)
        if url is None:
            saved[doc_type] = None
            continue
        filename = url.split("/")[-1]
        out_path = out_dir / filename
        if out_path.exists():
            logger.info("  [cached] %s", filename)
            saved[doc_type] = out_path
            continue
        try:
            resp = session.get(url)
            out_path.write_bytes(resp.content)
            logger.info("  [saved ] %s (%d KB)", filename, len(resp.content) // 1024)
            saved[doc_type] = out_path
        except Exception as e:
            logger.warning("  [fail  ] %s — %s", filename, e)
            saved[doc_type] = None

    return saved


def download_company_facts(cik: str, out_dir: Path, session: EdgarSession) -> Optional[Path]:
    """
    Download the EDGAR XBRL company-facts JSON for a CIK.
    This gives ALL XBRL facts across ALL filings in one shot — fastest path.
    """
    padded = cik.zfill(10)
    url = EDGAR_COMPANY_FACTS.format(cik=padded)
    out_path = out_dir / f"company_facts_{padded}.json"
    if out_path.exists():
        logger.info("  [cached] company_facts_%s.json", padded)
        return out_path
    try:
        resp = session.get(url, headers={"Host": "data.sec.gov"})
        out_path.write_bytes(resp.content)
        logger.info("  [saved ] company_facts_%s.json (%d KB)", padded, len(resp.content) // 1024)
        return out_path
    except Exception as e:
        logger.warning("  [fail  ] company_facts_%s — %s", padded, e)
        return None


# ---------------------------------------------------------------------------
# Live streaming — EDGAR RSS feed watcher
# ---------------------------------------------------------------------------

def stream_new_10k_filings(
    session: EdgarSession,
    poll_interval_seconds: int = 60,
) -> Iterator[dict]:
    """
    Poll the EDGAR RSS feed for new 10-K filings.
    Yields metadata dicts as new filings appear.

    Usage:
        for filing in stream_new_10k_filings(session):
            print(filing)
    """
    import xml.etree.ElementTree as ET

    seen_ids: set[str] = set()
    logger.info("Watching EDGAR for new 10-K filings (polling every %ds)...", poll_interval_seconds)

    while True:
        try:
            resp = session.get(EDGAR_EFTS_RSS)
            root = ET.fromstring(resp.text)

            ns = {"atom": "http://www.w3.org/2005/Atom"}
            entries = root.findall(".//atom:entry", ns)

            for entry in entries:
                entry_id = entry.findtext("atom:id", "", ns)
                if entry_id in seen_ids:
                    continue
                seen_ids.add(entry_id)

                title   = entry.findtext("atom:title", "", ns)
                updated = entry.findtext("atom:updated", "", ns)
                link    = entry.find("atom:link", ns)
                href    = link.get("href", "") if link is not None else ""

                # Parse CIK and accession from URL
                m = re.search(r"/data/(\d+)/(\d+)/", href)
                cik = m.group(1) if m else ""
                acc_nodash = m.group(2) if m else ""
                accession = f"{acc_nodash[:10]}-{acc_nodash[10:12]}-{acc_nodash[12:]}" if len(acc_nodash) == 18 else acc_nodash

                meta = {
                    "filing_id": f"{cik}/{accession}",
                    "cik": cik,
                    "accession": accession,
                    "title": title,
                    "updated": updated,
                    "url": href,
                }
                logger.info("NEW 10-K: %s", title)
                yield meta

        except KeyboardInterrupt:
            logger.info("Stopping watcher.")
            return
        except Exception as e:
            logger.warning("RSS fetch error: %s — retrying in %ds", e, poll_interval_seconds)

        time.sleep(poll_interval_seconds)


# ---------------------------------------------------------------------------
# EDGAR Full-Text Search (EFTS) — search by date range or keyword
# ---------------------------------------------------------------------------

def search_10k_filings(
    session: EdgarSession,
    start_date: str = "2023-01-01",
    end_date: str = "2023-12-31",
    max_results: int = 40,
) -> list[dict]:
    """
    Search EDGAR EFTS for 10-K filings in a date range.
    Returns list of {cik, accession, filed_date, company_name} dicts.
    """
    url = (
        f"https://efts.sec.gov/LATEST/search-index?"
        f"q=%2210-K%22&forms=10-K"
        f"&dateRange=custom&startdt={start_date}&enddt={end_date}"
        f"&_source=period_of_report,file_date,entity_name,file_num,period_of_report"
        f"&from=0&size={min(max_results, 100)}"
    )
    try:
        data = session.get_json(url)
        hits = data.get("hits", {}).get("hits", [])
        results = []
        for hit in hits:
            src = hit.get("_source", {})
            results.append({
                "cik": hit.get("_id", "").split(":")[0],
                "accession": hit.get("_id", ""),
                "filed_date": src.get("file_date"),
                "company_name": src.get("entity_name"),
                "period": src.get("period_of_report"),
            })
        return results
    except Exception as e:
        logger.error("EFTS search failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# Main sample downloader
# ---------------------------------------------------------------------------

def download_sample_corpus(
    out_dir: Path = FILINGS_DIR,
    max_per_company: int = 2,
    user_agent: str = DEFAULT_HEADERS["User-Agent"],
) -> list[dict]:
    """
    Download the sample corpus of 20 companies × up to 2 10-Ks each.
    Also downloads company-facts JSON (fast XBRL path) for each CIK.
    Returns a data_registry list compatible with EdgarExtractionEnv.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    session = EdgarSession(user_agent=user_agent)
    registry = []

    for cik, ticker, name in SAMPLE_TICKERS:
        logger.info("\n=== %s (%s) CIK=%s ===", name, ticker, cik)
        cik_dir = out_dir / cik.zfill(10)
        cik_dir.mkdir(exist_ok=True)

        # Download company-facts JSON (fastest XBRL source)
        facts_path = download_company_facts(cik, cik_dir, session)

        # Get submission history
        try:
            subs = get_submissions(cik, session)
        except Exception as e:
            logger.warning("  Could not fetch submissions: %s", e)
            continue

        company_name = subs.get("name", name)
        sic_code = subs.get("sic")

        # Download individual filings
        filings_meta = get_10k_accessions(subs, max_filings=max_per_company)
        for fm in filings_meta:
            accession = fm["accession"]
            acc_dir = cik_dir / accession
            acc_dir.mkdir(exist_ok=True)

            logger.info("  Downloading %s (filed %s)...", accession, fm["filed_date"])
            saved = download_filing(cik, accession, acc_dir, session)

            # Use company-facts JSON as XBRL path if raw XBRL not found
            xbrl_path = saved.get("xbrl") or facts_path

            entry = {
                "filing_id": f"{cik.zfill(10)}/{accession}",
                "cik": cik.zfill(10),
                "accession": accession,
                "ticker": ticker,
                "company_name": company_name,
                "sic_code": int(sic_code) if sic_code else None,
                "fiscal_year": int(fm["report_date"][:4]) if fm.get("report_date") else None,
                "fiscal_year_end": fm.get("report_date"),
                "filed_date": fm.get("filed_date"),
                "xbrl_path": str(xbrl_path) if xbrl_path else None,
                "html_path": str(saved["html"]) if saved.get("html") else None,
                "txt_path": str(saved["txt"]) if saved.get("txt") else None,
                "facts_json_path": str(facts_path) if facts_path else None,
            }
            registry.append(entry)

    # Save registry to JSON
    registry_path = out_dir / "registry.json"
    registry_path.write_text(json.dumps(registry, indent=2))
    logger.info("\nSaved registry with %d filings to %s", len(registry), registry_path)
    return registry


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDGAR 10-K downloader")
    parser.add_argument("--sample",   action="store_true", help="Download 20-company sample corpus")
    parser.add_argument("--watch",    action="store_true", help="Stream new 10-K filings live")
    parser.add_argument("--cik",      type=str, help="Download 10-Ks for a specific CIK")
    parser.add_argument("--ticker",   type=str, help="Resolve ticker → CIK and download")
    parser.add_argument("--max",      type=int, default=3, help="Max filings per company (default 3)")
    parser.add_argument("--out-dir",  type=str, default=str(FILINGS_DIR), help="Output directory")
    parser.add_argument("--user-agent", type=str, default=DEFAULT_HEADERS["User-Agent"])
    args = parser.parse_args()

    session = EdgarSession(user_agent=args.user_agent)
    out_dir = Path(args.out_dir)

    if args.sample:
        download_sample_corpus(out_dir=out_dir, max_per_company=args.max, user_agent=args.user_agent)

    elif args.watch:
        for filing_meta in stream_new_10k_filings(session):
            print(json.dumps(filing_meta, indent=2))

    elif args.ticker:
        cik = resolve_ticker_to_cik(args.ticker, session)
        if cik is None:
            print(f"Could not resolve ticker: {args.ticker}")
        else:
            print(f"Resolved {args.ticker} → CIK {cik}")
            subs = get_submissions(cik, session)
            filings = get_10k_accessions(subs, max_filings=args.max)
            cik_dir = out_dir / cik.zfill(10)
            cik_dir.mkdir(parents=True, exist_ok=True)
            download_company_facts(cik, cik_dir, session)
            for fm in filings:
                acc_dir = cik_dir / fm["accession"]
                acc_dir.mkdir(exist_ok=True)
                download_filing(cik, fm["accession"], acc_dir, session)

    elif args.cik:
        cik = args.cik.lstrip("0") or "0"
        subs = get_submissions(cik, session)
        filings = get_10k_accessions(subs, max_filings=args.max)
        cik_dir = out_dir / cik.zfill(10)
        cik_dir.mkdir(parents=True, exist_ok=True)
        download_company_facts(cik, cik_dir, session)
        for fm in filings:
            acc_dir = cik_dir / fm["accession"]
            acc_dir.mkdir(exist_ok=True)
            download_filing(cik, fm["accession"], acc_dir, session)

    else:
        parser.print_help()
