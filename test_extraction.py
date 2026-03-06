"""
test_extraction.py — Interactive extraction viewer.

Run directly:
    python test_extraction.py

Shows extractions from every available filing using all extractors,
so you can see exactly what each one returns and compare them.
"""

import json
import time
from pathlib import Path

# ── Colour helpers ────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
DIM    = "\033[2m"

def green(s):  return f"{GREEN}{s}{RESET}"
def yellow(s): return f"{YELLOW}{s}{RESET}"
def red(s):    return f"{RED}{s}{RESET}"
def cyan(s):   return f"{CYAN}{s}{RESET}"
def bold(s):   return f"{BOLD}{s}{RESET}"
def dim(s):    return f"{DIM}{s}{RESET}"

def fmt_value(v):
    if v is None:
        return dim("—")
    if abs(v) >= 1e9:
        return f"${v/1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"${v/1e6:.2f}M"
    if abs(v) >= 1e3:
        return f"${v/1e3:.2f}K"
    return f"{v:.4f}"

def fmt_conf(c):
    if c >= 0.8:  return green(f"{c:.2f}")
    if c >= 0.5:  return yellow(f"{c:.2f}")
    return red(f"{c:.2f}")

def fmt_reward(r):
    if r >= 0.8:  return green(f"{r:+.2f}")
    if r >= 0.0:  return yellow(f"{r:+.2f}")
    return red(f"{r:+.2f}")

# ── Load registry ─────────────────────────────────────────────────────────────

REGISTRY_PATH = Path("filings/registry.json")

def load_registry():
    if not REGISTRY_PATH.exists():
        print(red("No registry found. Run: python download_filings.py --sample"))
        exit(1)
    with open(REGISTRY_PATH) as f:
        return json.load(f)

# ── Menu helpers ──────────────────────────────────────────────────────────────

def pick_filing(registry):
    print(bold("\nAvailable filings:\n"))
    by_ticker = {}
    for entry in registry:
        by_ticker.setdefault(entry["ticker"], []).append(entry)

    indexed = []
    for ticker, entries in sorted(by_ticker.items()):
        for e in entries:
            indexed.append(e)
            yr = e.get("fiscal_year", "?")
            print(f"  {len(indexed):>3}.  {ticker:<6}  FY{yr}  {dim(e['filing_id'])}")

    print()
    choice = input("Pick a filing number (or press Enter for AAPL FY2025): ").strip()
    if not choice:
        return indexed[0]
    return indexed[int(choice) - 1]


def pick_field():
    fields = [
        "revenue", "cogs", "gross_profit", "operating_income", "net_income",
        "total_expenses", "interest_expense", "income_tax_expense",
        "eps_basic", "eps_diluted", "rd_expense", "sga_expense",
    ]
    print(bold("\nAvailable fields:\n"))
    for i, f in enumerate(fields, 1):
        print(f"  {i:>3}.  {f}")
    print(f"  {len(fields)+1:>3}.  ALL fields")
    print()
    choice = input("Pick a field number (or press Enter for ALL): ").strip()
    if not choice or int(choice) == len(fields) + 1:
        return fields
    return [fields[int(choice) - 1]]


def pick_extractor():
    extractors = [
        ("1", "XBRL direct  (ground truth)"),
        ("2", "HTML table"),
        ("3", "Regex on TXT"),
        ("4", "Claude LLM"),
        ("5", "ALL extractors + comparison"),
    ]
    print(bold("\nExtractors:\n"))
    for key, label in extractors:
        print(f"  {key}.  {label}")
    print()
    choice = input("Pick extractor (or press Enter for ALL): ").strip() or "5"
    return choice

# ── Individual extractors ─────────────────────────────────────────────────────

def run_xbrl(entry, field):
    from xbrl_extractor import load_xbrl_from_facts_file, extract_field
    facts_path = Path(entry["facts_json_path"])
    filing = load_xbrl_from_facts_file(
        facts_path,
        cik=entry["cik"],
        accession=entry["accession"],
        fiscal_year=entry.get("fiscal_year"),
    )
    if filing is None:
        return None, 0.0, "could not load XBRL"
    r = extract_field(filing, field)
    return r.value, r.confidence, r.xbrl_tag or r.error or "—"


def run_html(entry, field):
    from html_extractor import extract_from_html
    html_path = entry.get("html_path")
    if not html_path or not Path(html_path).exists():
        return None, 0.0, "no HTML file"
    value, conf = extract_from_html(Path(html_path), field)
    return value, conf, Path(html_path).name


def run_regex(entry, field):
    from regex_extractor import extract_from_txt
    txt_path = entry.get("txt_path")
    if not txt_path or not Path(txt_path).exists():
        return None, 0.0, "no TXT file"
    value, conf = extract_from_txt(Path(txt_path), field)
    return value, conf, Path(txt_path).name


def run_llm(entry, field):
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None, 0.0, "ANTHROPIC_API_KEY not set"
    from llm_extractor import extract_with_claude
    doc_path = Path(entry["html_path"]) if entry.get("html_path") else Path(entry["txt_path"])
    if not doc_path.exists():
        return None, 0.0, "no document"
    t0 = time.perf_counter()
    value, conf = extract_with_claude(doc_path, field)
    latency = time.perf_counter() - t0
    return value, conf, f"claude  ({latency:.1f}s)"

# ── Display ───────────────────────────────────────────────────────────────────

def print_single_extractor(entry, fields, extractor_fn, label):
    from xbrl_extractor import compute_reward

    # Get ground truth for reward scoring
    gt = {}
    from xbrl_extractor import load_xbrl_from_facts_file, extract_field
    filing = load_xbrl_from_facts_file(
        Path(entry["facts_json_path"]),
        cik=entry["cik"],
        accession=entry["accession"],
    )

    print()
    print(bold(f"  {'Field':<22} {'Value':>12}  {'Conf':>5}  {'Source'}"))
    print("  " + "─" * 70)

    for field in fields:
        gt_result = extract_field(filing, field) if filing else None
        ground_truth = gt_result.value if gt_result else None

        value, conf, source = extractor_fn(entry, field)
        reward = compute_reward(value, ground_truth, action_index=0)

        val_str  = fmt_value(value)
        conf_str = fmt_conf(conf)
        rew_str  = fmt_reward(reward)

        print(f"  {field:<22} {val_str:>12}  {conf_str}  {dim(str(source)[:45])}  {rew_str}")


def print_all_extractors(entry, fields):
    from xbrl_extractor import load_xbrl_from_facts_file, extract_field, compute_reward
    from dotenv import load_dotenv
    load_dotenv()

    filing = load_xbrl_from_facts_file(
        Path(entry["facts_json_path"]),
        cik=entry["cik"],
        accession=entry["accession"],
    )

    extractor_defs = [
        ("XBRL",  run_xbrl,  0),
        ("HTML",  run_html,  1),
        ("Regex", run_regex, 2),
        ("LLM",   run_llm,   3),
    ]

    for field in fields:
        gt_result = extract_field(filing, field) if filing else None
        ground_truth = gt_result.value if gt_result else None

        print()
        print(bold(f"  {field}") + (f"  {dim('(ground truth: ' + fmt_value(ground_truth) + ')')}" if ground_truth else ""))
        print(f"  {'Extractor':<8}  {'Value':>12}  {'Conf':>5}  {'Reward':>7}  Source")
        print("  " + "─" * 65)

        for label, fn, action_idx in extractor_defs:
            value, conf, source = fn(entry, field)
            reward = compute_reward(value, ground_truth, action_index=action_idx)
            val_str  = fmt_value(value)
            conf_str = fmt_conf(conf)
            rew_str  = fmt_reward(reward)
            print(f"  {label:<8}  {val_str:>12}  {conf_str}  {rew_str}  {dim(str(source)[:40])}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    from dotenv import load_dotenv
    load_dotenv()

    print(bold(cyan("\n╔══════════════════════════════════════╗")))
    print(bold(cyan("║     edgaRL — Extraction Viewer       ║")))
    print(bold(cyan("╚══════════════════════════════════════╝")))

    registry = load_registry()

    while True:
        entry = pick_filing(registry)
        fields = pick_field()
        extractor = pick_extractor()

        ticker = entry.get("ticker", "?")
        fy     = entry.get("fiscal_year", "?")

        print()
        print(bold(f"  {ticker}  FY{fy}  —  {entry['filing_id']}"))
        print("  " + "═" * 70)

        if extractor == "1":
            print_single_extractor(entry, fields, run_xbrl,  "XBRL")
        elif extractor == "2":
            print_single_extractor(entry, fields, run_html,  "HTML")
        elif extractor == "3":
            print_single_extractor(entry, fields, run_regex, "Regex")
        elif extractor == "4":
            print_single_extractor(entry, fields, run_llm,   "LLM")
        else:
            print_all_extractors(entry, fields)

        print()
        again = input(dim("  Test another? [Y/n]: ")).strip().lower()
        if again == "n":
            break

    print(bold(cyan("\n  Done.\n")))


if __name__ == "__main__":
    main()
