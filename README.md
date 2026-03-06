# edgaRL

A reinforcement learning system that learns *how* to extract financial data from SEC 10-K filings — not just what to extract.

---

## The Problem

Extracting structured financials from 10-K filings is harder than it looks. Every company formats its income statement differently. XBRL tags are inconsistently applied. Some filers use clean HTML tables, others bury numbers in dense prose. No single extractor works across all companies and all fields.

The standard fix is a hard-coded cascade of fallbacks. This system learns that cascade instead.

---

## How It Works

Each extraction is framed as an RL episode:

- **State** — which field, what formats are available, what prior attempts returned
- **Action** — which extractor to try: XBRL, HTML table, regex, Claude LLM, or arithmetic derivation
- **Reward** — how close the result is to the XBRL ground truth

The policy learns, per field and per industry sector, which extractor to reach for first — and when to fall back.

---

## Target Fields

12 income statement fields from US-GAAP filings:

| Field | XBRL Tag |
|---|---|
| Revenue | `RevenueFromContractWithCustomerExcludingAssessedTax` |
| COGS | `CostOfRevenue` |
| Gross Profit | `GrossProfit` |
| Operating Income | `OperatingIncomeLoss` |
| Net Income | `NetIncomeLoss` |
| Total Expenses | `OperatingExpenses` |
| Interest Expense | `InterestExpense` |
| Income Tax Expense | `IncomeTaxExpenseBenefit` |
| EPS Basic | `EarningsPerShareBasic` |
| EPS Diluted | `EarningsPerShareDiluted` |
| R&D Expense | `ResearchAndDevelopmentExpense` |
| SG&A Expense | `SellingGeneralAndAdministrativeExpense` |

---

## Architecture

```
SEC EDGAR
    │
    ▼
Filing Ingestion          download_filings.py
(XBRL · HTML · TXT)       verifies 10-K via SGML header before saving
    │
    ▼
Observation Builder       env.py
                          32-dim state: field one-hot, format flags,
                          last attempt result, filing metadata, action priors
    │
    ▼
Policy (UCB1 Bandit)      bandit.py
                          one bandit per field × sector, upgradeable to PPO
    │
    ▼
Action Space
  ├── XBRL direct         xbrl_extractor.py  — ground truth oracle path
  ├── HTML table          html_extractor.py  — income statement table parser
  ├── Regex on TXT        regex_extractor.py — proximity search near labels
  ├── Claude LLM          llm_extractor.py   — structured JSON prompt fallback
  ├── Derived value       env.py             — e.g. Gross Profit = Revenue − COGS
  └── Flag missing
    │
    ▼
Reward                    xbrl_extractor.py
  +1.0  exact match
  +0.8  within 1%
  +0.3  correct magnitude
  −0.5  null / missing
  −1.0  type mismatch
    │
    ▼
DuckDB                    edgar_rl.duckdb
                          logs every episode, step, and extracted value
```

---

## Quickstart

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate

# 2. Dependencies
pip install sec-edgar-downloader requests beautifulsoup4 lxml pandas \
            anthropic gymnasium stable-baselines3 duckdb python-dotenv \
            tqdm arelle-release

# 3. API key — only needed for LLM fallback (action 3)
echo "ANTHROPIC_API_KEY=your-key-here" > .env

# 4. Download sample corpus — 20 companies × 3 years each
python download_filings.py --sample

# 5. Train
python train.py --episodes 2000 --bandit context

# 6. Evaluate
python train.py --eval
```

---

## Commands

| Command | What it does |
|---|---|
| `python train.py --episodes 2000 --bandit context` | Train the bandit from scratch |
| `python train.py --episodes 1000 --bandit context` | Resume from last checkpoint |
| `python train.py --ppo --timesteps 100000` | Upgrade to PPO after bandit converges |
| `python train.py --eval` | Evaluate trained policy, print per-field accuracy |
| `python xbrl_extractor.py <filing.xml>` | Run ground truth oracle on one filing |
| `python xbrl_extractor.py --test-reward` | Sanity check the reward function |
| `python html_extractor.py <filing.htm> <field>` | Test HTML extractor on one field |
| `python regex_extractor.py <filing.txt> <field>` | Test regex extractor on one field |
| `python llm_extractor.py <filing.htm>` | Extract all 12 fields via Claude |
| `python download_filings.py --ticker TSLA` | Add a company to the corpus |
| `python download_filings.py --watch` | Stream new 10-K filings live from EDGAR |

---

## Data Sources

All data is from SEC EDGAR public APIs — no API key required.

| Source | URL | Used for |
|---|---|---|
| Company facts (XBRL) | `data.sec.gov/api/xbrl/companyfacts/` | Ground truth, bulk XBRL |
| Submission history | `data.sec.gov/submissions/` | Filing index per company |
| Filing archives | `sec.gov/Archives/edgar/data/` | Raw XBRL, HTML, TXT |
| Live RSS feed | `sec.gov/cgi-bin/browse-edgar?type=10-K&output=atom` | Real-time new filings |

SEC enforces a 10 req/s rate limit. The downloader respects this automatically.

---

## Training Results

2000 episodes · 55 filings · 20 companies across 5 sectors

| Episodes | Mean Reward |
|---|---|
| 500 | +0.48 |
| 1000 | +0.71 |
| 1500 | +0.70 |
| 2000 | +0.83 |
| **Final 100** | **+0.90** |

The bandit converges to `xbrl_direct` for ~90% of episodes, using HTML/regex/LLM selectively for filings where XBRL tags are absent or ambiguous.

---

## Project Structure

```
edgaRL/
├── field_config.json      Target fields, XBRL tag aliases, reward weights
├── xbrl_extractor.py      Ground truth oracle + reward function
├── html_extractor.py      HTML income statement table parser
├── regex_extractor.py     Regex extraction from raw TXT filings
├── llm_extractor.py       Claude-based LLM fallback extractor
├── env.py                 Gymnasium RL environment
├── bandit.py              UCB1 bandit policy (FieldBandit, ContextBandit)
├── train.py               Training loop (bandit + PPO) and evaluation
├── db.py                  DuckDB schema and logging helpers
└── download_filings.py    EDGAR downloader and live RSS watcher
```

---

## Roadmap

- [ ] Hand-verified golden set across 10 filings for calibrated accuracy measurement
- [ ] HTML and regex tuning to reach >95% accuracy on all 12 fields
- [ ] PPO training on top of converged bandit
- [ ] Edge cases: restated filings, fiscal year mismatches, foreign private issuers
- [ ] Accuracy breakdown by SIC sector and filer size
