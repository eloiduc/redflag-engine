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

# ── Professional CSS — minimal, dark, no decoration ──────────────────────────
st.markdown("""
<style>
/* Hide Streamlit chrome */
#MainMenu          { visibility: hidden; }
footer             { visibility: hidden; }
header             { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* Base font */
html, body, [class*="css"] {
    font-family: "Inter", "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0c0c0c;
    border-right: 1px solid #1c1c1c;
}
[data-testid="stSidebar"] * {
    font-size: 12px;
    letter-spacing: 0.03em;
}

/* Remove radio button label uppercase Streamlit adds */
[data-testid="stSidebar"] label {
    text-transform: uppercase;
    font-size: 11px;
    color: #666 !important;
    letter-spacing: 0.08em;
}

/* Metric tiles */
[data-testid="stMetric"] {
    background-color: #111;
    border: 1px solid #1e1e1e;
    padding: 14px 18px;
}
[data-testid="stMetricLabel"] {
    font-size: 10px !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #555 !important;
}
[data-testid="stMetricValue"] {
    font-size: 20px !important;
    font-weight: 500;
    color: #e0e0e0 !important;
}

/* Primary button */
.stButton > button[kind="primary"] {
    background-color: #1a1a1a;
    border: 1px solid #333;
    color: #ccc;
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    border-radius: 2px;
}
.stButton > button[kind="primary"]:hover {
    background-color: #222;
    border-color: #555;
    color: #fff;
}

/* Secondary / default button */
.stButton > button {
    background-color: transparent;
    border: 1px solid #2a2a2a;
    color: #888;
    font-size: 11px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    border-radius: 2px;
}
.stButton > button:hover {
    border-color: #444;
    color: #bbb;
    background-color: #111;
}

/* Form submit button */
[data-testid="stFormSubmitButton"] > button {
    background-color: #111;
    border: 1px solid #333;
    color: #bbb;
    font-size: 11px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    border-radius: 2px;
    width: 100%;
}
[data-testid="stFormSubmitButton"] > button:hover {
    background-color: #1a1a1a;
    border-color: #555;
    color: #fff;
}

/* Input fields */
[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] select {
    background-color: #0f0f0f;
    border: 1px solid #222;
    color: #ccc;
    font-size: 12px;
    border-radius: 2px;
}

/* Divider */
hr {
    border-color: #1c1c1c;
    margin: 16px 0;
}

/* Headers */
h1, h2, h3 {
    font-weight: 500;
    letter-spacing: -0.01em;
    color: #d0d0d0;
}
h1 { font-size: 18px; }
h2 { font-size: 14px; text-transform: uppercase; letter-spacing: 0.06em; color: #999; }
h3 { font-size: 13px; }

/* Alert / info / success / error */
[data-testid="stAlert"] {
    border-radius: 2px;
    font-size: 12px;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #1c1c1c;
}

/* Caption */
.stCaption {
    color: #444 !important;
    font-size: 11px !important;
}

/* Expander */
[data-testid="stExpander"] {
    border: 1px solid #1c1c1c;
    border-radius: 2px;
}

/* Container border */
[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid #1c1c1c !important;
    border-radius: 2px;
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
    st.markdown("**RED FLAG ENGINE**")
    st.caption("Earnings Call Analysis")
    st.divider()
    page = st.radio("Navigation", _PAGES, key="page", label_visibility="collapsed")
    st.divider()
    st.caption("Claude · Streamlit")


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

    reports = sorted(_OUTPUTS.glob("*.md"))
    if not reports:
        st.info("No reports. Run an analysis first.")
        return

    for row_start in range(0, len(reports), 3):
        cols = st.columns(3)
        for col, md_path in zip(cols, reports[row_start : row_start + 3]):
            meta = _parse_report_meta(md_path)
            with col:
                with st.container(border=True):
                    st.subheader(meta["company"])
                    st.caption(f"{meta['now']} vs {meta['prev']}")
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
    with st.expander("📈 Signal Validation (Backtest)", expanded=False):
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

        with st.spinner("Running analysis..."):
            report_path = run_pipeline(
                company     = company,
                now_period  = now_period,
                prev_period = prev_period,
                now_path    = str(now_save_path),
                prev_path   = str(prev_save_path),
            )

        st.success("Analysis complete.")
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
    disc_m = re.search(r"> .*?\*\*Disclaimer:\*\* (.+?)(?=\n\n|\Z)", hdr, re.DOTALL)
    if disc_m:
        st.warning(disc_m.group(1).strip())

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
        with st.expander("⚠️ Abandoned Metrics", expanded=True):
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
        with st.expander("📊 Hedging Intensity", expanded=True):
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

    # ── Peer & Supplier Signals ───────────────────────────────────────────────
    ps_sec = sections.get("Peer & Supplier Signals", "")
    if ps_sec:
        with st.expander("🌐 Peer & Supplier Signals", expanded=False):
            rows = _parse_md_table(ps_sec)
            if rows:
                df_ps = pd.DataFrame(rows)
                st.dataframe(df_ps, use_container_width=True, hide_index=True)

    st.divider()

    # ── Monitor Checklist ────────────────────────────────────────────────────
    chk_sec = sections.get("Monitor Checklist", "")
    if chk_sec:
        st.subheader("Monitor Checklist")
        items = re.findall(r"- \[ \] (.+)", chk_sec)
        for i, item in enumerate(items):
            st.checkbox(item, value=False, key=f"chk_{selected}_{i}")
        st.caption("Verify each item against the source transcript.")

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

    # ── Backtest Context ──────────────────────────────────────────────────────
    bt_sec = sections.get("Backtest Context", "")
    if bt_sec:
        with st.expander("📈 Backtest Context", expanded=False):
            rows = _parse_md_table(bt_sec)
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            disclaimer_m = re.search(r"\*([^*]+)\*", bt_sec)
            if disclaimer_m:
                st.caption(disclaimer_m.group(1).strip())

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
