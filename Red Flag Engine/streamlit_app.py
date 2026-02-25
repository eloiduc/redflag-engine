"""streamlit_app.py — Red Flag Engine web interface.

Run locally:
    streamlit run streamlit_app.py

Deploy to Streamlit Community Cloud:
    1. Push repo to GitHub (.env and .streamlit/secrets.toml must be gitignored)
    2. In the Cloud dashboard, add ANTHROPIC_API_KEY to Secrets
    3. Set main file path to: streamlit_app.py
"""

from __future__ import annotations

import os
import re
import sys
import traceback
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Project root anchor (this file lives at the project root) ────────────────
_APP_ROOT = Path(__file__).resolve().parent
_OUTPUTS  = _APP_ROOT / "outputs"
_DATA     = _APP_ROOT / "data"
_SRC      = _APP_ROOT / "src"

# Ensure src/ is importable so run_pipeline can find ingest, segment, etc.
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ── Page config — must be the first Streamlit call ───────────────────────────
st.set_page_config(
    page_title="Red Flag Engine",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Institutional CSS — Bloomberg-style dark terminal ─────────────────────────
st.markdown("""
<style>
/* ── Reset & base ─────────────────────────────────────────────────────────── */
#MainMenu, footer, header, [data-testid="stDecoration"] { display: none !important; }

html, body, [class*="css"] {
    font-family: "Inter", "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
    background-color: #080808;
    color: #c8c8c8;
}

/* ── Main content area ───────────────────────────────────────────────────── */
.stApp, .main, [data-testid="stAppViewContainer"] {
    background-color: #080808;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #060606;
    border-right: 1px solid #181818;
}
[data-testid="stSidebar"] * {
    font-size: 11px;
    letter-spacing: 0.04em;
}
[data-testid="stSidebar"] label {
    text-transform: uppercase;
    font-size: 10px;
    color: #444 !important;
    letter-spacing: 0.12em;
}

/* ── Typography ──────────────────────────────────────────────────────────── */
h1 {
    font-size: 16px;
    font-weight: 600;
    letter-spacing: 0.02em;
    color: #e8e8e8;
    text-transform: uppercase;
    border-bottom: 1px solid #1a1a1a;
    padding-bottom: 8px;
    margin-bottom: 16px;
}
h2 {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #555;
    margin-top: 20px;
    margin-bottom: 8px;
}
h3 {
    font-size: 12px;
    font-weight: 500;
    color: #999;
    letter-spacing: 0.04em;
}

/* ── Metric tiles ────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background-color: #0d0d0d;
    border: 1px solid #1a1a1a;
    border-top: 2px solid #1e1e1e;
    padding: 12px 16px;
}
[data-testid="stMetricLabel"] {
    font-size: 9px !important;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: #444 !important;
}
[data-testid="stMetricValue"] {
    font-size: 22px !important;
    font-weight: 400;
    color: #ddd !important;
    font-variant-numeric: tabular-nums;
}
[data-testid="stMetricDelta"] {
    font-size: 11px !important;
}

/* ── Buttons ─────────────────────────────────────────────────────────────── */
.stButton > button[kind="primary"],
.stButton > button {
    background-color: #0d0d0d;
    border: 1px solid #252525;
    color: #999;
    font-size: 10px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    border-radius: 1px;
    transition: all 0.1s ease;
}
.stButton > button[kind="primary"]:hover,
.stButton > button:hover {
    background-color: #141414;
    border-color: #3a3a3a;
    color: #e0e0e0;
}

/* ── Form submit ─────────────────────────────────────────────────────────── */
[data-testid="stFormSubmitButton"] > button {
    background-color: #0f1620;
    border: 1px solid #1e3050;
    color: #6a9fd8;
    font-size: 10px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    border-radius: 1px;
    width: 100%;
}
[data-testid="stFormSubmitButton"] > button:hover {
    background-color: #141d2d;
    border-color: #2a4a80;
    color: #8ab8e8;
}

/* ── Inputs ──────────────────────────────────────────────────────────────── */
[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] select,
[data-testid="stFileUploader"] {
    background-color: #0a0a0a;
    border: 1px solid #1e1e1e;
    color: #bbb;
    font-size: 12px;
    border-radius: 1px;
}

/* ── Divider ─────────────────────────────────────────────────────────────── */
hr {
    border: none;
    border-top: 1px solid #141414;
    margin: 20px 0;
}

/* ── Alerts ──────────────────────────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 1px;
    font-size: 12px;
    border-left-width: 3px;
}

/* ── Dataframe ───────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid #1a1a1a;
}
[data-testid="stDataFrame"] table {
    font-size: 12px;
}

/* ── Caption ─────────────────────────────────────────────────────────────── */
.stCaption, [data-testid="stCaptionContainer"] {
    color: #3a3a3a !important;
    font-size: 10px !important;
    letter-spacing: 0.04em;
}

/* ── Expander ────────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #141414;
    border-left: 2px solid #1e1e1e;
    border-radius: 1px;
    background-color: #0a0a0a;
}
[data-testid="stExpander"] summary {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #555;
}
[data-testid="stExpander"] summary:hover {
    color: #888;
}

/* ── Container border ────────────────────────────────────────────────────── */
[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid #141414 !important;
    border-radius: 1px;
    background-color: #0a0a0a;
}

/* ── Multiselect ─────────────────────────────────────────────────────────── */
[data-testid="stMultiSelect"] {
    font-size: 12px;
}

/* ── Download button ─────────────────────────────────────────────────────── */
[data-testid="stDownloadButton"] > button {
    background-color: transparent;
    border: 1px solid #1e1e1e;
    color: #555;
    font-size: 10px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    border-radius: 1px;
}
[data-testid="stDownloadButton"] > button:hover {
    border-color: #333;
    color: #888;
}

/* ── Spinner ─────────────────────────────────────────────────────────────── */
[data-testid="stSpinner"] {
    font-size: 11px;
    color: #444 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
</style>
""", unsafe_allow_html=True)

# ── Ensure outputs/ and data/ exist at startup ───────────────────────────────
os.makedirs(_OUTPUTS, exist_ok=True)
os.makedirs(_DATA,    exist_ok=True)

# ── API key: st.secrets (Cloud) → os.environ (local .env) ───────────────────
_api_key: str = ""
try:
    _api_key = st.secrets["ANTHROPIC_API_KEY"]
except (KeyError, FileNotFoundError):
    _api_key = os.environ.get("ANTHROPIC_API_KEY", "")

if not _api_key:
    st.error(
        "ANTHROPIC_API_KEY is not set.  \n"
        "Local: add it to `.streamlit/secrets.toml`  \n"
        "Cloud: paste it in the Streamlit Community Cloud secrets panel"
    )
    st.stop()

# Propagate key so src.main's anthropic.Anthropic() client picks it up.
os.environ["ANTHROPIC_API_KEY"] = _api_key

# ── Sidebar navigation ───────────────────────────────────────────────────────
_PAGES = ["Overview", "New Analysis", "Report"]

# Transfer any pending programmatic navigation BEFORE the radio widget renders.
# (Streamlit forbids writing session_state[key] after the widget owning that
#  key has been instantiated, so we stage the target in "_nav_to" and apply it
#  here, at the very top of each re-run.)
if "_nav_to" in st.session_state:
    st.session_state["page"] = st.session_state.pop("_nav_to")

with st.sidebar:
    st.markdown(
        "<div style='font-size:13px;font-weight:700;letter-spacing:0.12em;"
        "text-transform:uppercase;color:#ccc;padding:4px 0 2px'>Red Flag Engine</div>"
        "<div style='font-size:10px;letter-spacing:0.08em;color:#3a3a3a;padding-bottom:4px'>"
        "Earnings Intelligence</div>",
        unsafe_allow_html=True,
    )
    st.divider()
    page = st.radio("Navigation", _PAGES, key="page", label_visibility="collapsed")
    st.divider()
    st.caption("Powered by Claude")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Shared helpers                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

_FILENAME_RE = re.compile(r"^([A-Z0-9]+)_([^_]+)_vs_([^_]+)\.md$", re.IGNORECASE)


def _parse_report_meta(md_path: Path) -> dict:
    """Return dashboard metadata for one report file."""
    m       = _FILENAME_RE.match(md_path.name)
    company = m.group(1).upper() if m else md_path.stem
    now     = m.group(2)         if m else "—"
    prev    = m.group(3)         if m else "—"

    content = md_path.read_text(encoding="utf-8", errors="ignore")
    total_m = re.search(r"\| Total changes detected \| (\d+) \|", content)
    high_m  = re.search(r"\| High / Critical \| \*\*(\d+)\*\* \|", content)

    return {
        "company":       company,
        "now":           now,
        "prev":          prev,
        "total_changes": int(total_m.group(1)) if total_m else 0,
        "high_critical": int(high_m.group(1))  if high_m  else 0,
        "filename":      md_path.name,
    }


def _split_sections(content: str) -> dict[str, str]:
    """Split a report into named sections using the --- separators."""
    parts    = re.split(r"\n\n---\n\n", content)
    sections: dict[str, str] = {}
    for part in parts:
        stripped = part.strip()
        if stripped.startswith("# Red Flag Report"):
            sections["header"] = stripped
        else:
            hm = re.match(r"^## (.+)", stripped)
            if hm:
                sections[hm.group(1).strip()] = stripped
    return sections


def _split_pipe_row(line: str) -> list[str]:
    """Split a Markdown table row on unescaped pipes; strip leading/trailing empties."""
    parts = re.split(r"(?<!\\)\|", line)
    while parts and not parts[0].strip():
        parts.pop(0)
    while parts and not parts[-1].strip():
        parts.pop()
    return [p.strip().replace("\\|", "|") for p in parts]


def _parse_md_table(text: str) -> list[dict[str, str]]:
    """Parse a Markdown pipe table into a list of row dicts."""
    lines = [ln for ln in text.splitlines() if re.match(r"^\s*\|", ln)]
    if len(lines) < 3:
        return []
    headers = _split_pipe_row(lines[0])
    rows: list[dict[str, str]] = []
    for line in lines[2:]:
        cells = _split_pipe_row(line)
        if len(cells) == len(headers):
            rows.append(dict(zip(headers, cells)))
    return rows


def _sev_style(val: str) -> str:
    return {
        "Critical":      "background-color: #4a0000; color: #ffaaaa",
        "High":          "background-color: #3a1800; color: #ffcc99",
        "Medium":        "background-color: #2a2000; color: #ffe680",
        "Low":           "background-color: #1a1a1a; color: #888888",
        "Informational": "background-color: #111111; color: #555555",
    }.get(val, "")


def _change_style(val: str) -> str:
    if "WORSENED"  in val: return "background-color: #4a0000; color: #ffaaaa"
    if "NEW"       in val: return "background-color: #001a40; color: #99bbff"
    if "IMPROVED"  in val: return "background-color: #001a0d; color: #66cc88"
    if "UNCHANGED" in val: return "background-color: #111111; color: #555555"
    return ""


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 1 — Overview                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def page_dashboard() -> None:
    st.header("Overview")

    reports = sorted(_OUTPUTS.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not reports:
        st.info("No reports found. Run a new analysis to get started.")
        return

    for row_start in range(0, len(reports), 3):
        cols = st.columns(3)
        for col, md_path in zip(cols, reports[row_start : row_start + 3]):
            meta = _parse_report_meta(md_path)
            with col:
                with st.container(border=True):
                    st.markdown(
                        f"<div style='font-size:15px;font-weight:600;letter-spacing:0.04em;"
                        f"color:#ddd'>{meta['company']}</div>"
                        f"<div style='font-size:10px;letter-spacing:0.06em;color:#3a3a3a;"
                        f"text-transform:uppercase;margin-bottom:10px'>"
                        f"{meta['now']} vs {meta['prev']}</div>",
                        unsafe_allow_html=True,
                    )
                    col_a, col_b = st.columns(2)
                    col_a.metric("Changes", meta["total_changes"])
                    col_b.metric("High / Critical", meta["high_critical"])
                    if st.button(
                        "Open Report",
                        key=f"view_{meta['filename']}",
                        use_container_width=True,
                    ):
                        st.session_state["selected_report"] = meta["filename"]
                        st.session_state["_nav_to"] = "Report"
                        st.rerun()

    st.divider()

    # ── Signal Validation (Backtest) ─────────────────────────────────────────
    with st.expander("Signal Validation", expanded=False):
        try:
            from backtest import load_backtest_summary
            dates_path = _APP_ROOT / "earnings_dates.json"
            df_bt = load_backtest_summary(_OUTPUTS, dates_path)
            if hasattr(df_bt, "empty") and df_bt.empty:
                st.info(
                    "No backtest data available.  "
                    "Populate `earnings_dates.json` at the project root to enable."
                )
            else:
                n = len(df_bt)
                st.dataframe(df_bt, use_container_width=True, hide_index=True)
                if n < 20:
                    st.caption(
                        f"n={n} — insufficient sample size for statistical inference. "
                        "Retrospective data only — not a trading signal."
                    )
        except Exception as exc:
            st.info(f"Signal validation unavailable: {exc}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 2 — New Analysis                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def page_generate() -> None:
    st.header("New Analysis")
    st.caption("Upload two earnings call transcripts to generate a comparative red-flag report.")

    with st.form("pipeline_form"):
        company = st.text_input(
            "Ticker",
            placeholder="BA",
            max_chars=10,
        )
        col1, col2 = st.columns(2)
        now_period  = col1.text_input("Current quarter", placeholder="2025Q4")
        prev_period = col2.text_input("Prior quarter",   placeholder="2025Q3")

        now_file = st.file_uploader(
            "Current quarter transcript (.txt or .pdf)",
            type=["txt", "pdf"],
        )
        prev_file = st.file_uploader(
            "Prior quarter transcript (.txt or .pdf)",
            type=["txt", "pdf"],
        )
        submitted = st.form_submit_button(
            "Run Analysis",
            use_container_width=True,
            type="primary",
        )

    if not submitted:
        return

    # ── Validation ───────────────────────────────────────────────────────────
    _period_re = re.compile(r"^\d{4}Q[1-4]$")
    errors: list[str] = []

    if not company.strip():
        errors.append("Ticker is required.")
    if not now_period.strip():
        errors.append("Current quarter is required.")
    elif not _period_re.match(now_period.strip()):
        errors.append("Current quarter must match format YYYYQ# (e.g. 2025Q4).")
    if not prev_period.strip():
        errors.append("Prior quarter is required.")
    elif not _period_re.match(prev_period.strip()):
        errors.append("Prior quarter must match format YYYYQ# (e.g. 2025Q3).")
    if now_file is None:
        errors.append("Current quarter transcript is required.")
    if prev_file is None:
        errors.append("Prior quarter transcript is required.")

    if errors:
        for err in errors:
            st.error(err)
        return

    company     = company.strip().upper()
    now_period  = now_period.strip()
    prev_period = prev_period.strip()

    # ── Save uploaded files to data/<COMPANY>/ ───────────────────────────────
    company_dir = _DATA / company
    company_dir.mkdir(parents=True, exist_ok=True)

    now_ext  = Path(now_file.name).suffix.lower()
    prev_ext = Path(prev_file.name).suffix.lower()

    now_save_path  = company_dir / f"{now_period}_transcript{now_ext}"
    prev_save_path = company_dir / f"{prev_period}_transcript{prev_ext}"

    now_save_path.write_bytes(now_file.read())
    prev_save_path.write_bytes(prev_file.read())

    # ── Run pipeline ─────────────────────────────────────────────────────────
    try:
        from src.main import run_pipeline  # deferred import; resolves src/ path

        with st.spinner("Analysing transcripts…"):
            report_path = run_pipeline(
                company     = company,
                now_period  = now_period,
                prev_period = prev_period,
                now_path    = str(now_save_path),
                prev_path   = str(prev_save_path),
            )

        st.session_state["selected_report"] = os.path.basename(report_path)
        st.session_state["_nav_to"] = "Report"
        st.rerun()

    except Exception as exc:
        st.error(f"Pipeline failed: {exc}")
        st.code(traceback.format_exc())


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 3 — Report                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def page_view() -> None:
    selected = st.session_state.get("selected_report")
    if not selected:
        st.info("Select a report from the Overview first.")
        st.stop()

    report_path = _OUTPUTS / selected
    if not report_path.exists():
        st.error(f"Report not found: {selected}")
        st.stop()

    content  = report_path.read_text(encoding="utf-8", errors="ignore")
    sections = _split_sections(content)

    # ── Header metrics ───────────────────────────────────────────────────────
    hdr        = sections.get("header", "")
    company_m  = re.search(r"\| Company \| \*\*(.+?)\*\* \|",           hdr)
    now_m      = re.search(r"\| Current quarter \| `(.+?)` \|",          hdr)
    prev_m     = re.search(r"\| Prior quarter \| `(.+?)` \|",            hdr)
    ts_m       = re.search(r"\| Generated \| (.+?) \|",                  hdr)
    total_m    = re.search(r"\| Total changes detected \| (\d+) \|",     hdr)
    high_m     = re.search(r"\| High / Critical \| \*\*(\d+)\*\* \|",   hdr)

    company_val   = company_m.group(1) if company_m else selected
    now_val       = now_m.group(1)     if now_m     else "—"
    prev_val      = prev_m.group(1)    if prev_m    else "—"
    ts_val        = ts_m.group(1)      if ts_m      else "—"
    total_changes = int(total_m.group(1)) if total_m else 0
    high_critical = int(high_m.group(1))  if high_m  else 0

    st.header(f"{company_val} — Earnings Analysis")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ticker",          company_val)
    col2.metric("Current Quarter", now_val)
    col3.metric("Prior Quarter",   prev_val)
    col4.metric("High / Critical", high_critical)
    st.caption(f"Generated {ts_val}  ·  {total_changes} changes detected")

    # ── Disclaimer ───────────────────────────────────────────────────────────
    disc_m = re.search(r"> \*\*Disclaimer:\*\* (.+?)(?=\n\n|\Z)", hdr, re.DOTALL)
    if disc_m:
        st.caption(f"Triage aid only — {disc_m.group(1).strip()}")

    st.divider()

    # ── Executive Summary ────────────────────────────────────────────────────
    exec_sec = sections.get("Executive Summary", "")
    if exec_sec:
        st.subheader("Executive Summary")
        bullets = re.findall(r"^- (.+)$", exec_sec, re.MULTILINE)
        for bullet in bullets:
            if "WORSENED" in bullet:
                st.error(bullet)
            elif "NEW" in bullet:
                st.info(bullet)
            elif "IMPROVED" in bullet:
                st.success(bullet)
            else:
                st.warning(bullet)

    st.divider()

    # ── Red Flags Table ──────────────────────────────────────────────────────
    rf_sec = sections.get("Red Flags", "")
    st.subheader("Findings")
    if rf_sec:
        rows = _parse_md_table(rf_sec)
        rows = [r for r in rows if r.get("#", "—") != "—"]

        if not rows:
            st.info("No material changes detected.")
        else:
            df = pd.DataFrame(rows)

            all_sevs = ["Critical", "High", "Medium", "Low", "Informational"]
            all_cats = sorted(df["Category"].unique().tolist()) if "Category" in df.columns else []

            fc1, fc2 = st.columns(2)
            sel_sev = fc1.multiselect(
                "Severity",
                options=all_sevs,
                default=all_sevs,
            )
            sel_cat = fc2.multiselect(
                "Category",
                options=all_cats,
                default=all_cats,
            )

            if "Sev" in df.columns:
                df = df[df["Sev"].isin(sel_sev)]
            if "Category" in df.columns and sel_cat:
                df = df[df["Category"].isin(sel_cat)]

            if df.empty:
                st.info("No rows match the selected filters.")
            else:
                styled = df.style
                if "Sev" in df.columns:
                    styled = styled.map(_sev_style,    subset=["Sev"])
                if "Change" in df.columns:
                    styled = styled.map(_change_style, subset=["Change"])

                col_cfg: dict = {}
                if "Evidence (Now)" in df.columns:
                    col_cfg["Evidence (Now)"]  = st.column_config.TextColumn(width="large")
                if "Evidence (Prev)" in df.columns:
                    col_cfg["Evidence (Prev)"] = st.column_config.TextColumn(width="large")

                st.dataframe(styled, use_container_width=True, column_config=col_cfg)
    else:
        st.info("Findings section not found in this report.")

    st.divider()

    # ── Abandoned Metrics ─────────────────────────────────────────────────────
    ab_sec = sections.get("Abandoned Metrics", "")
    if ab_sec:
        with st.expander("Abandoned Metrics", expanded=True):
            rows = _parse_md_table(ab_sec)
            if rows:
                df_ab = pd.DataFrame(rows)
                st.dataframe(df_ab, use_container_width=True, hide_index=True)
            else:
                # Render the prose description if there's no table
                body = "\n".join(
                    ln for ln in ab_sec.splitlines()
                    if not ln.startswith("## Abandoned Metrics")
                ).strip()
                st.markdown(body)

    # ── Hedging Intensity ─────────────────────────────────────────────────────
    hg_sec = sections.get("Hedging Intensity", "")
    if hg_sec:
        with st.expander("Hedging Intensity", expanded=True):
            rows = _parse_md_table(hg_sec)
            if rows:
                df_hg = pd.DataFrame(rows)

                def _flag_row_style(row: pd.Series) -> list[str]:
                    flag_val = row.get("Flag", "")
                    if flag_val and flag_val.strip():
                        return ["background-color: #3d1a1a"] * len(row)
                    return [""] * len(row)

                styled_hg = df_hg.style.apply(_flag_row_style, axis=1)
                st.dataframe(styled_hg, use_container_width=True, hide_index=True)
            else:
                # Fall back to raw markdown if table parsing fails
                body = "\n".join(
                    ln for ln in hg_sec.splitlines()
                    if not ln.startswith("## Hedging Intensity")
                ).strip()
                if body:
                    st.markdown(body)
                else:
                    st.caption("No hedging intensity data for this report.")

    # ── Peer & Supplier Signals ───────────────────────────────────────────────
    ps_sec = sections.get("Peer & Supplier Signals", "")
    if ps_sec:
        with st.expander("Peer & Supplier Signals", expanded=False):
            rows = _parse_md_table(ps_sec)
            if rows:
                df_ps = pd.DataFrame(rows)
                st.dataframe(df_ps, use_container_width=True, hide_index=True)

    st.divider()

    # ── AI Announcement Sensitivity ──────────────────────────────────────────
    ai_sec = sections.get("AI Announcement Sensitivity", "")
    if ai_sec:
        st.divider()
        # Extract and display the sensitivity level as a top-line metric
        level_m = re.search(
            r"\*\*Sensitivity Level:\*\*\s*(CRITICAL|HIGH|MEDIUM|LOW|MINIMAL)",
            ai_sec,
        )
        direction_m = re.search(
            r"\*\*AI Exposure Direction:\*\*\s*([^\n]+)",
            ai_sec,
        )
        level_color = {
            "CRITICAL": "error",
            "HIGH":     "warning",
            "MEDIUM":   "warning",
            "LOW":      "info",
            "MINIMAL":  "info",
        }
        if level_m:
            level = level_m.group(1)
            direction = direction_m.group(1).strip() if direction_m else ""
            col_lv, col_dir = st.columns([1, 3])
            col_lv.metric("AI Sensitivity", level)
            if direction:
                col_dir.caption(direction)
            fn = getattr(st, level_color.get(level, "info"))
            if level in ("CRITICAL", "HIGH"):
                fn(f"This stock carries **{level}** sensitivity to AI announcements.")
        # Render the full section body (strip the first two header lines already shown above)
        body_lines = ai_sec.splitlines()
        # Drop the ## heading line and the two metadata bold lines for a clean body
        body = "\n".join(
            ln for ln in body_lines
            if not ln.startswith("## AI Announcement Sensitivity")
            and not ln.startswith("**Sensitivity Level:**")
            and not ln.startswith("**AI Exposure Direction:**")
        ).strip()
        st.markdown(body)

    # ── Disruption Lag Analysis ───────────────────────────────────────────────
    dl_sec = sections.get("Disruption Lag Analysis", "")
    if dl_sec:
        st.divider()
        st.subheader("Disruption Lag Analysis")

        # Extract overall score and management awareness
        score_m = re.search(
            r"\*\*Disruption Lag Score:\*\*\s*(CRITICAL|HIGH|MEDIUM|LOW|MINIMAL)",
            dl_sec,
        )
        aware_m = re.search(
            r"\*\*Management AI Awareness:\*\*\s*(\w+)",
            dl_sec,
        )

        _dl_score_color = {
            "CRITICAL": "error",
            "HIGH":     "error",
            "MEDIUM":   "warning",
            "LOW":      "info",
            "MINIMAL":  "info",
        }

        if score_m:
            dl_score    = score_m.group(1)
            dl_aware    = aware_m.group(1) if aware_m else "—"
            col_sc, col_aw, col_sp = st.columns([1, 1, 2])
            col_sc.metric("Disruption Lag", dl_score)
            col_aw.metric("Mgmt. Awareness", dl_aware.title())
            fn = getattr(st, _dl_score_color.get(dl_score, "info"))
            if dl_score in ("CRITICAL", "HIGH"):
                fn(
                    f"**{dl_score} disruption lag detected** — one or more core "
                    "business functions appear economically replicable by currently "
                    "available AI and have not yet been repriced by the market."
                )

        # Signals table
        sig_block = re.search(
            r"### Signals Detected\n(.*?)(?=###|\Z)", dl_sec, re.DOTALL
        )
        if sig_block:
            rows = _parse_md_table(sig_block.group(1))
            if rows:
                df_dl = pd.DataFrame(rows)

                # Colour rows by Lag column value
                def _lag_row_style(row: pd.Series) -> list[str]:
                    lag_cell = row.get("Lag", "")
                    if "CRITICAL" in lag_cell:
                        return ["background-color: #4a0000; color: #ffaaaa"] * len(row)
                    if "HIGH" in lag_cell:
                        return ["background-color: #3a1800; color: #ffcc99"] * len(row)
                    if "MEDIUM" in lag_cell:
                        return ["background-color: #2a2000; color: #ffe680"] * len(row)
                    return [""] * len(row)

                styled_dl = df_dl.style.apply(_lag_row_style, axis=1)
                col_cfg_dl: dict = {}
                for col in ("Moat Claim", "AI Capability", "Best Analogue"):
                    if col in df_dl.columns:
                        col_cfg_dl[col] = st.column_config.TextColumn(width="large")
                st.dataframe(
                    styled_dl,
                    use_container_width=True,
                    hide_index=True,
                    column_config=col_cfg_dl,
                )

        # Signal details in an expander
        detail_block = re.search(
            r"### Signal Details\n(.*?)(?=_Disruption lag analysis|\Z)", dl_sec, re.DOTALL
        )
        if detail_block and detail_block.group(1).strip():
            with st.expander("Signal Details", expanded=False):
                st.markdown(detail_block.group(1).strip())

        # Disclaimer / footnote
        footnote_m = re.search(r"_Disruption lag analysis[^_]+_", dl_sec)
        if footnote_m:
            st.caption(footnote_m.group(0).strip("_"))

    # ── Prediction Market Context ─────────────────────────────────────────────
    pm_sec = sections.get("Prediction Market Context", "")
    if pm_sec:
        st.divider()
        st.subheader("Prediction Market Context")

        # Surface CONTRADICTS count as a top-level alert
        contra_count = len(re.findall(r"\*\*CONTRADICTS\*\*", pm_sec))
        if contra_count:
            st.error(
                f"{contra_count} prediction market signal(s) contradict management guidance. "
                "Review the cross-reference table below."
            )

        # Active Markets table
        # The section has two sub-tables under ### Active Markets and ### Cross-Reference
        active_block = re.search(
            r"### Active Markets\n(.*?)(?=###|\Z)", pm_sec, re.DOTALL
        )
        crossref_block = re.search(
            r"### Cross-Reference with Management Claims\n(.*?)(?=###|\Z)", pm_sec, re.DOTALL
        )

        if active_block:
            st.caption("Active Markets")
            rows = _parse_md_table(active_block.group(1))
            if rows:
                df_pm = pd.DataFrame(rows)
                # Strip Markdown link syntax for display: [text](url) → text
                if "Market" in df_pm.columns:
                    df_pm["Market"] = df_pm["Market"].str.replace(
                        r"\[([^\]]+)\]\([^)]+\)", r"\1", regex=True
                    )
                st.dataframe(df_pm, use_container_width=True, hide_index=True)

        if crossref_block:
            st.caption("Cross-Reference with Management Claims")
            rows = _parse_md_table(crossref_block.group(1))
            if rows:
                df_cr = pd.DataFrame(rows)

                # Strip link syntax
                if "Market" in df_cr.columns:
                    df_cr["Market"] = df_cr["Market"].str.replace(
                        r"\[([^\]]+)\]\([^)]+\)", r"\1", regex=True
                    )
                # Strip bold markers from Alignment column
                if "Alignment" in df_cr.columns:
                    df_cr["Alignment"] = df_cr["Alignment"].str.replace(
                        r"\*\*([^*]+)\*\*", r"\1", regex=True
                    )

                def _align_style(row: pd.Series) -> list[str]:
                    a = row.get("Alignment", "")
                    if "CONTRADICTS" in a:
                        return ["background-color: #4a0000; color: #ffaaaa"] * len(row)
                    if "CONFIRMS" in a:
                        return ["background-color: #001a0d; color: #66cc88"] * len(row)
                    return [""] * len(row)

                styled_cr = df_cr.style.apply(_align_style, axis=1)
                col_cfg: dict = {}
                for col in ("Claim", "Interpretation"):
                    if col in df_cr.columns:
                        col_cfg[col] = st.column_config.TextColumn(width="large")
                st.dataframe(
                    styled_cr,
                    use_container_width=True,
                    hide_index=True,
                    column_config=col_cfg,
                )

        st.caption(
            "Source: Polymarket / Kalshi. Prediction markets reflect aggregate public "
            "belief backed by real capital, not analyst consensus. CONTRADICTS signals "
            "indicate the market is pricing a materially different outcome than management guidance."
        )
    else:
        st.divider()
        st.caption(
            "Prediction Market Context — no active markets found for this company "
            "(Polymarket API may be unreachable, or no liquid markets exist for this ticker)."
        )

    # ── Backtest Context ──────────────────────────────────────────────────────
    bt_sec = sections.get("Backtest Context", "")
    if bt_sec:
        with st.expander("Backtest Context", expanded=False):
            rows = _parse_md_table(bt_sec)
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            disclaimer_m = re.search(r"\*([^*]+)\*", bt_sec)
            if disclaimer_m:
                st.caption(disclaimer_m.group(1).strip())
    else:
        st.caption(
            "Backtest Context — no earnings date configured for this report. "
            "Add an entry to earnings_dates.json to enable post-earnings return tracking."
        )

    # ── Limitations & Methodology ────────────────────────────────────────────
    lim_sec  = sections.get("Limitations", "")
    meth_sec = sections.get("Methodology", "")
    combined = "\n\n".join(s for s in [lim_sec, meth_sec] if s)
    if combined:
        with st.expander("Limitations & Methodology", expanded=False):
            st.markdown(combined)

    st.divider()

    # ── Download ──────────────────────────────────────────────────────────────
    with open(report_path, encoding="utf-8") as fh:
        raw_content = fh.read()

    st.download_button(
        label="Export Report (.md)",
        data=raw_content,
        file_name=selected,
        mime="text/markdown",
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Router                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

if page == "Overview":
    page_dashboard()
elif page == "New Analysis":
    page_generate()
elif page == "Report":
    page_view()
