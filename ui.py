"""
ui.py — edgaRL Dashboard
Run: .venv/bin/streamlit run ui.py
"""

import json
import pickle
import time
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="edgaRL",
    page_icon="▣",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — full dark override, no white iframes
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&display=swap');

/* ── Global reset ─────────────────────────────────────────────────────────── */
*, html, body {
    font-family: 'IBM Plex Mono', 'Courier New', monospace !important;
}
.stApp, .main, section.main, div[data-testid="stAppViewContainer"],
div[data-testid="stMainBlockContainer"] {
    background-color: #0a0a0a !important;
    color: #d4d4d4 !important;
}

/* Remove top padding / Streamlit chrome */
header[data-testid="stHeader"] { display: none !important; }
div[data-testid="stToolbar"]   { display: none !important; }
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    max-width: 100% !important;
}

/* ── Sidebar ──────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background-color: #080808 !important;
    border-right: 1px solid #1a1a1a !important;
}
section[data-testid="stSidebar"] * { color: #c8c8c8 !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stRadio label {
    color: #444 !important;
    font-size: 9px !important;
    letter-spacing: 0.18em;
    text-transform: uppercase;
}

/* ── Tabs ─────────────────────────────────────────────────────────────────── */
div[data-testid="stTabs"] {
    background: transparent !important;
}
.stTabs [role="tablist"] {
    background: transparent !important;
    border-bottom: 1px solid #1e1e1e !important;
    gap: 0 !important;
}
.stTabs [role="tab"] {
    background: transparent !important;
    color: #3a3a3a !important;
    font-size: 9px !important;
    font-weight: 500 !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    padding: 10px 22px !important;
    border: none !important;
    border-radius: 0 !important;
}
.stTabs [role="tab"]:hover { color: #888 !important; }
.stTabs [aria-selected="true"] {
    color: #ffffff !important;
    border-bottom: 1px solid #ffffff !important;
    background: transparent !important;
}

/* ── Metrics ──────────────────────────────────────────────────────────────── */
div[data-testid="metric-container"] {
    background: #0d0d0d !important;
    border: 1px solid #1a1a1a !important;
    border-radius: 0 !important;
    padding: 18px 20px !important;
}
div[data-testid="metric-container"] label,
div[data-testid="metric-container"] [data-testid="stMetricLabel"] p {
    color: #444 !important;
    font-size: 9px !important;
    letter-spacing: 0.16em !important;
    text-transform: uppercase !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 18px !important;
    font-weight: 500 !important;
}

/* ── Selectbox ────────────────────────────────────────────────────────────── */
.stSelectbox > div > div,
.stSelectbox > div > div > div {
    background: #0d0d0d !important;
    border: 1px solid #1e1e1e !important;
    border-radius: 0 !important;
    color: #d4d4d4 !important;
}

/* ── Button ───────────────────────────────────────────────────────────────── */
.stButton > button {
    width: 100% !important;
    background: #ffffff !important;
    color: #000000 !important;
    border: none !important;
    border-radius: 0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 9px !important;
    font-weight: 600 !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    padding: 12px 0 !important;
}
.stButton > button:hover { background: #cccccc !important; }

/* ── Spinner ──────────────────────────────────────────────────────────────── */
div[data-testid="stSpinner"] { color: #666 !important; }

/* ── General text ─────────────────────────────────────────────────────────── */
p, li, span, div { color: #d4d4d4; }
h1, h2, h3 {
    color: #ffffff !important;
    letter-spacing: 0.05em;
    font-weight: 500;
}
hr { border-color: #1a1a1a !important; }
code, pre {
    background: #0d0d0d !important;
    color: #888 !important;
    border: 1px solid #1e1e1e !important;
    border-radius: 0 !important;
    font-size: 10px !important;
}

/* ── Custom HTML table (our own, not st.dataframe) ───────────────────────── */
.rl-table-wrap {
    overflow-x: auto;
    border: 1px solid #1a1a1a;
    margin-top: 4px;
}
.rl-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11.5px;
    background: #0a0a0a;
    color: #d4d4d4;
}
.rl-table thead tr {
    background: #080808;
    border-bottom: 1px solid #242424;
}
.rl-table thead th {
    padding: 11px 18px;
    text-align: left;
    font-size: 9px;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #444 !important;
    white-space: nowrap;
}
.rl-table thead th.num { text-align: right; }
.rl-table tbody tr {
    border-bottom: 1px solid #111;
    transition: background 0.1s;
}
.rl-table tbody tr:hover { background: #0f0f0f; }
.rl-table tbody td {
    padding: 10px 18px;
    vertical-align: middle;
    color: #c8c8c8;
}
.rl-table tbody td.field-name {
    color: #888;
    font-size: 10.5px;
    letter-spacing: 0.04em;
}
.rl-table tbody td.num {
    text-align: right;
    font-variant-numeric: tabular-nums;
    color: #ffffff;
}
.rl-table tbody td.score-pass {
    text-align: right;
    color: #ffffff;
    font-weight: 500;
}
.rl-table tbody td.score-partial {
    text-align: right;
    color: #555;
}
.rl-table tbody td.score-fail {
    text-align: right;
    color: #2e2e2e;
}
.rl-table tbody td.score-nogt {
    text-align: right;
    color: #252525;
    font-style: italic;
}
.rl-table tbody td.dash { color: #2a2a2a; }
.rl-table tbody tr.row-pass td.field-name { color: #888; }
.rl-table tbody tr.row-fail td.field-name { color: #4a4a4a; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Constants & helpers
# ─────────────────────────────────────────────────────────────────────────────

REGISTRY_PATH  = Path("filings/registry.json")
CHECKPOINT_DIR = Path("checkpoints")
DB_PATH        = Path("edgar_rl.duckdb")

ALL_FIELDS = [
    "revenue", "cogs", "gross_profit", "operating_income", "net_income",
    "total_expenses", "interest_expense", "income_tax_expense",
    "eps_basic", "eps_diluted", "rd_expense", "sga_expense",
]

FIELD_LABELS = {
    "revenue":           "Revenue",
    "cogs":              "Cost of Revenue",
    "gross_profit":      "Gross Profit",
    "operating_income":  "Operating Income",
    "net_income":        "Net Income",
    "total_expenses":    "Total Expenses",
    "interest_expense":  "Interest Expense",
    "income_tax_expense":"Income Tax Expense",
    "eps_basic":         "EPS  Basic",
    "eps_diluted":       "EPS  Diluted",
    "rd_expense":        "R&D Expense",
    "sga_expense":       "SG&A Expense",
}

ACTION_NAMES = ["XBRL", "HTML", "Regex", "LLM", "Derived", "Missing"]


@st.cache_data
def load_registry():
    if not REGISTRY_PATH.exists():
        return []
    return json.loads(REGISTRY_PATH.read_text())


def fmt_val(v):
    if v is None:
        return None
    av = abs(v)
    if av == 0:
        return "$0"
    if av >= 1e9:
        return f"${v/1e9:.2f}B"
    if av >= 1e6:
        return f"${v/1e6:.2f}M"
    if av >= 1e3:
        return f"${v/1e3:.1f}K"
    return f"{v:.4f}"


def score_cell(rwd, has_gt):
    """Return (html_content, td_class) for a reward value."""
    if not has_gt:
        return '<span style="color:#222">no gt</span>', "score-nogt"
    if rwd is None:
        return "—", "dash"
    if rwd >= 0.8:
        return f"+{rwd:.2f}", "score-pass"
    if rwd >= 0.0:
        return f"{rwd:+.2f}", "score-partial"
    return f"{rwd:+.2f}", "score-fail"


def build_extraction_table(results: dict) -> str:
    """Render extraction results as a styled HTML table."""
    rows_html = ""
    for field in ALL_FIELDS:
        r       = results[field]
        gt      = r["gt"]
        has_gt  = gt is not None

        gt_str  = fmt_val(gt)   or '<span class="dash">—</span>'
        h_str   = fmt_val(r["html_val"]) or '<span class="dash">—</span>'
        rx_str  = fmt_val(r["rgx_val"])  or '<span class="dash">—</span>'

        h_score_txt,  h_cls  = score_cell(r["html_rwd"],  has_gt)
        rx_score_txt, rx_cls = score_cell(r["rgx_rwd"],   has_gt)

        # Row class: dim the whole row if both extractors failed
        both_fail = has_gt and r["html_rwd"] < 0 and r["rgx_rwd"] < 0
        row_cls   = "row-fail" if both_fail else "row-pass"

        rows_html += f"""
        <tr class="{row_cls}">
          <td class="field-name">{FIELD_LABELS.get(field, field)}</td>
          <td class="num">{gt_str}</td>
          <td class="num">{h_str}</td>
          <td class="{h_cls}">{h_score_txt}</td>
          <td class="num">{rx_str}</td>
          <td class="{rx_cls}">{rx_score_txt}</td>
        </tr>"""

    return f"""
    <div class="rl-table-wrap">
      <table class="rl-table">
        <thead>
          <tr>
            <th>Field</th>
            <th class="num">Ground Truth</th>
            <th class="num">HTML</th>
            <th class="num">Score</th>
            <th class="num">Regex</th>
            <th class="num">Score</th>
          </tr>
        </thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>"""


def build_rl_table(field_bandits: dict) -> str:
    rows_html = ""
    for field in ALL_FIELDS:
        b = field_bandits.get(field)
        if b is None:
            continue
        Q = list(b.Q)
        N = list(b.N)
        best_idx  = Q.index(max(Q))
        best_name = ACTION_NAMES[best_idx] if best_idx < len(ACTION_NAMES) else "?"
        total_n   = sum(N)

        def qfmt(i):
            return f"{Q[i]:+.3f}" if i < len(Q) else "—"

        rows_html += f"""
        <tr>
          <td class="field-name">{FIELD_LABELS.get(field, field)}</td>
          <td class="num" style="color:#fff;font-weight:500">{best_name}</td>
          <td class="num score-pass">{qfmt(0)}</td>
          <td class="num score-partial">{qfmt(1)}</td>
          <td class="num score-partial">{qfmt(2)}</td>
          <td class="num score-partial">{qfmt(3)}</td>
          <td class="num score-partial">{qfmt(4)}</td>
          <td class="num dash">{total_n:,}</td>
        </tr>"""

    return f"""
    <div class="rl-table-wrap">
      <table class="rl-table">
        <thead>
          <tr>
            <th>Field</th>
            <th class="num">Best Action</th>
            <th class="num">Q(XBRL)</th>
            <th class="num">Q(HTML)</th>
            <th class="num">Q(Regex)</th>
            <th class="num">Q(LLM)</th>
            <th class="num">Q(Derived)</th>
            <th class="num">Pulls</th>
          </tr>
        </thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>"""


def build_corpus_table(registry: list) -> str:
    rows_html = ""
    for e in registry:
        hp = e.get("html_path", "")
        tp = e.get("txt_path", "")
        fp = e.get("facts_json_path", "")
        ok_html = "◈" if hp and Path(hp).exists() else '<span style="color:#222">○</span>'
        ok_txt  = "◈" if tp and Path(tp).exists() else '<span style="color:#222">○</span>'
        ok_xbrl = "◈" if fp and Path(fp).exists() else '<span style="color:#222">○</span>'
        rows_html += f"""
        <tr>
          <td style="color:#fff;font-weight:500">{e.get("ticker","?")}</td>
          <td class="field-name">{e.get("company","?")[:32]}</td>
          <td class="num">{e.get("fiscal_year","?")}</td>
          <td class="num">{ok_xbrl}</td>
          <td class="num">{ok_html}</td>
          <td class="num">{ok_txt}</td>
          <td class="field-name" style="font-size:9px">{e.get("filing_id","?")}</td>
        </tr>"""

    return f"""
    <div class="rl-table-wrap">
      <table class="rl-table">
        <thead>
          <tr>
            <th>Ticker</th>
            <th>Company</th>
            <th class="num">FY</th>
            <th class="num">XBRL</th>
            <th class="num">HTML</th>
            <th class="num">TXT</th>
            <th>Filing ID</th>
          </tr>
        </thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>"""


def run_all_extractors(entry):
    from xbrl_extractor import load_xbrl_from_facts_file, extract_field, compute_reward
    from html_extractor  import extract_from_html
    from regex_extractor import extract_from_txt

    filing = load_xbrl_from_facts_file(
        Path(entry["facts_json_path"]),
        cik=entry["cik"],
        accession=entry["accession"],
        fiscal_year=entry.get("fiscal_year"),
    )

    results = {}
    for field in ALL_FIELDS:
        gt_r = extract_field(filing, field) if filing else None
        gt   = gt_r.value if gt_r else None

        html_val = html_conf = None
        if entry.get("html_path") and Path(entry["html_path"]).exists():
            html_val, html_conf = extract_from_html(Path(entry["html_path"]), field)

        rgx_val = rgx_conf = None
        if entry.get("txt_path") and Path(entry["txt_path"]).exists():
            rgx_val, rgx_conf = extract_from_txt(Path(entry["txt_path"]), field)

        results[field] = {
            "gt":       gt,
            "html_val": html_val, "html_conf": html_conf,
            "html_rwd": compute_reward(html_val, gt, 1),
            "rgx_val":  rgx_val,  "rgx_conf":  rgx_conf,
            "rgx_rwd":  compute_reward(rgx_val,  gt, 2),
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

registry = load_registry()
if not registry:
    st.error("No registry found. Run:  python download_filings.py --sample")
    st.stop()

with st.sidebar:
    st.markdown(
        "<p style='font-size:18px;font-weight:600;color:#fff;letter-spacing:.06em;margin-bottom:2px'>EDGARL</p>"
        "<p style='font-size:8px;color:#333;letter-spacing:.2em;margin-top:0'>EXTRACTION DASHBOARD</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    by_ticker: dict[str, list] = {}
    for e in registry:
        by_ticker.setdefault(e.get("ticker", "?"), []).append(e)

    selected_ticker = st.selectbox("COMPANY", sorted(by_ticker.keys()))

    entries = sorted(
        by_ticker[selected_ticker],
        key=lambda x: x.get("fiscal_year", 0),
        reverse=True,
    )
    years = [e.get("fiscal_year", "?") for e in entries]
    selected_year = st.selectbox("FISCAL YEAR", years)

    entry = next(
        (e for e in entries if e.get("fiscal_year") == selected_year),
        entries[0],
    )

    st.markdown("---")
    run_btn = st.button("▶  RUN EXTRACTION")
    st.markdown(
        "<p style='font-size:8px;color:#222;margin-top:32px;letter-spacing:.1em'>edgaRL · SEC 10-K RL System</p>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["  EXTRACTION  ", "  RL PERFORMANCE  ", "  CORPUS  "])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Extraction
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    # Header
    st.markdown(
        f"<h2 style='margin-bottom:2px;font-size:22px'>{selected_ticker} &mdash; FY{selected_year}</h2>"
        f"<p style='font-size:9px;color:#333;letter-spacing:.12em;margin-top:0'>{entry.get('filing_id','')}</p>",
        unsafe_allow_html=True,
    )

    # File badges
    def badge(label, key):
        path = entry.get(key)
        ok   = path and Path(path).exists()
        col  = "#ffffff" if ok else "#222222"
        return f"<span style='font-size:9px;letter-spacing:.15em;color:{col};margin-right:24px'>{'◈' if ok else '○'}  {label}</span>"

    st.markdown(
        badge("XBRL", "facts_json_path") +
        badge("HTML", "html_path") +
        badge("TXT",  "txt_path"),
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='margin:16px 0'>", unsafe_allow_html=True)

    # Auto-run when company/year changes
    cache_key = f"{entry.get('filing_id')}_{selected_year}"
    if run_btn or st.session_state.get("last_key") != cache_key:
        st.session_state["last_key"] = cache_key
        with st.spinner("Running extractors..."):
            t0 = time.perf_counter()
            st.session_state["results"] = run_all_extractors(entry)
            st.session_state["elapsed"] = time.perf_counter() - t0

    results = st.session_state.get("results", {})
    elapsed = st.session_state.get("elapsed", 0.0)

    if results:
        st.markdown(build_extraction_table(results), unsafe_allow_html=True)

        # Summary metrics
        st.markdown("<hr style='margin:20px 0'>", unsafe_allow_html=True)
        html_hits = sum(1 for f in ALL_FIELDS if results[f]["gt"] and results[f]["html_rwd"] >= 0.8)
        rgx_hits  = sum(1 for f in ALL_FIELDS if results[f]["gt"] and results[f]["rgx_rwd"]  >= 0.8)
        scored    = sum(1 for f in ALL_FIELDS if results[f]["gt"] is not None)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("HTML Hit Rate",  f"{html_hits} / {scored}   ({100 * html_hits // max(scored,1)}%)")
        c2.metric("Regex Hit Rate", f"{rgx_hits} / {scored}   ({100 * rgx_hits  // max(scored,1)}%)")
        c3.metric("Extraction",     f"{elapsed:.2f}s")
        c4.metric("Accession",      entry.get("accession", "—")[:20])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RL Performance
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("<h2 style='font-size:16px;letter-spacing:.08em'>RL PERFORMANCE</h2>", unsafe_allow_html=True)
    st.markdown("<hr style='margin:12px 0'>", unsafe_allow_html=True)

    # ── Bandit Q-values ──────────────────────────────────────────────────────
    ckpt_files = sorted(CHECKPOINT_DIR.glob("bandit_*.pkl")) if CHECKPOINT_DIR.exists() else []

    if not ckpt_files:
        st.markdown(
            "<p style='color:#333;font-size:11px'>No checkpoint found.<br>"
            "Run:  <code>python train.py --episodes 2000 --bandit context</code></p>",
            unsafe_allow_html=True,
        )
    else:
        latest = ckpt_files[-1]
        st.markdown(f"<p style='font-size:9px;color:#333;letter-spacing:.1em'>CHECKPOINT  {latest.name}</p>", unsafe_allow_html=True)

        try:
            with open(latest, "rb") as f:
                bandit = pickle.load(f)

            field_bandits = {}
            if hasattr(bandit, "bandits"):
                field_bandits = bandit.bandits
            elif hasattr(bandit, "field_bandits"):
                field_bandits = next(iter(bandit.field_bandits.values()), {})

            if field_bandits:
                st.markdown(build_rl_table(field_bandits), unsafe_allow_html=True)
            else:
                st.markdown("<p style='color:#333'>No per-field bandit data found.</p>", unsafe_allow_html=True)

        except Exception as exc:
            st.markdown(f"<p style='color:#333;font-size:10px'>Could not load checkpoint: {exc}</p>", unsafe_allow_html=True)

    st.markdown("<hr style='margin:24px 0'>", unsafe_allow_html=True)

    # ── Training history from DuckDB ─────────────────────────────────────────
    st.markdown("<p style='font-size:9px;color:#333;letter-spacing:.18em'>TRAINING HISTORY</p>", unsafe_allow_html=True)

    if not DB_PATH.exists():
        st.markdown("<p style='color:#333;font-size:10px'>No training database. Run <code>python train.py</code> first.</p>", unsafe_allow_html=True)
    else:
        try:
            import duckdb
            import pandas as pd
            # Open, read all data, close immediately so the file lock is released
            # before training tries to open the DB for writing.
            conn = duckdb.connect(str(DB_PATH), read_only=True)
            tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
            df_ep = pd.DataFrame()
            if "episodes" in tables:
                df_ep = conn.execute(
                    "SELECT episode_id, field_name, reward, action_taken FROM episodes ORDER BY episode_id"
                ).df()
            conn.close()  # release lock immediately

            if not df_ep.empty:
                df_ep["rolling"] = df_ep["reward"].rolling(100, min_periods=1).mean()

                st.markdown("<p style='font-size:9px;color:#333;letter-spacing:.1em;margin-bottom:4px'>ROLLING AVG REWARD  (100-episode window)</p>", unsafe_allow_html=True)
                st.line_chart(
                    df_ep.set_index("episode_id")[["rolling"]],
                    height=200,
                    color=["#ffffff"],
                )

                st.markdown("<p style='font-size:9px;color:#333;letter-spacing:.1em;margin:16px 0 4px'>PER-FIELD ACCURACY</p>", unsafe_allow_html=True)

                stats = (
                    df_ep.groupby("field_name")
                    .agg(mean_reward=("reward","mean"), win_rate=("reward", lambda x: (x>=0.8).mean()), n=("episode_id","count"))
                    .reset_index()
                    .sort_values("mean_reward", ascending=False)
                )

                rows_html = ""
                for _, row in stats.iterrows():
                    wr_pct = f"{row.win_rate*100:.0f}%"
                    mr     = row.mean_reward
                    mr_cls = "score-pass" if mr >= 0.8 else ("score-partial" if mr >= 0.3 else "score-fail")
                    rows_html += f"""
                    <tr>
                      <td class="field-name">{FIELD_LABELS.get(row.field_name, row.field_name)}</td>
                      <td class="num {mr_cls}">{mr:+.3f}</td>
                      <td class="num" style="color:#fff">{wr_pct}</td>
                      <td class="num dash">{int(row.n):,}</td>
                    </tr>"""

                st.markdown(f"""
                <div class="rl-table-wrap">
                  <table class="rl-table">
                    <thead>
                      <tr>
                        <th>Field</th>
                        <th class="num">Mean Reward</th>
                        <th class="num">Win Rate</th>
                        <th class="num">Episodes</th>
                      </tr>
                    </thead>
                    <tbody>{rows_html}</tbody>
                  </table>
                </div>""", unsafe_allow_html=True)

                st.markdown("<hr style='margin:20px 0'>", unsafe_allow_html=True)
                total_eps   = df_ep["episode_id"].nunique()
                overall_win = (df_ep["reward"] >= 0.8).mean()
                last100_win = (df_ep.tail(100)["reward"] >= 0.8).mean()

                ma, mb, mc = st.columns(3)
                ma.metric("Total Episodes",    f"{total_eps:,}")
                mb.metric("Overall Win Rate",  f"{overall_win*100:.1f}%")
                mc.metric("Last 100 Win Rate", f"{last100_win*100:.1f}%")

        except Exception as exc:
            st.markdown(f"<p style='color:#333;font-size:10px'>DB error: {exc}</p>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Corpus
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("<h2 style='font-size:16px;letter-spacing:.08em'>CORPUS</h2>", unsafe_allow_html=True)
    st.markdown("<hr style='margin:12px 0'>", unsafe_allow_html=True)

    st.markdown(build_corpus_table(registry), unsafe_allow_html=True)

    st.markdown("<hr style='margin:20px 0'>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Filings", len(registry))
    c2.metric("Companies",     len(by_ticker))
    c3.metric("With HTML",     sum(1 for e in registry if e.get("html_path") and Path(e["html_path"]).exists()))
