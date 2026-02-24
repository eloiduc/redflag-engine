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
    page_icon="🚩",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
        "**ANTHROPIC_API_KEY is not set.**  \n"
        "• **Local:** add it to `.streamlit/secrets.toml`  \n"
        "• **Cloud:** paste it in the Streamlit Community Cloud secrets panel  \n"
        "• **CLI only:** add `ANTHROPIC_API_KEY=sk-ant-...` to `.env`"
    )
    st.stop()

# Propagate key so src.main's anthropic.Anthropic() client picks it up.
os.environ["ANTHROPIC_API_KEY"] = _api_key

# ── Sidebar navigation ───────────────────────────────────────────────────────
_PAGES = ["📊 Dashboard", "📤 Generate Report", "📄 View Report"]

# Transfer any pending programmatic navigation BEFORE the radio widget renders.
# (Streamlit forbids writing session_state[key] after the widget owning that
#  key has been instantiated, so we stage the target in "_nav_to" and apply it
#  here, at the very top of each re-run.)
if "_nav_to" in st.session_state:
    st.session_state["page"] = st.session_state.pop("_nav_to")

with st.sidebar:
    st.title("🚩 Red Flag Engine")
    st.caption("Earnings Call Monitor")
    st.divider()
    page = st.radio("Navigation", _PAGES, key="page", label_visibility="collapsed")
    st.divider()
    st.caption("Powered by Claude · Built with Streamlit")


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
    # Remove the empty strings that flank the outer |…|
    while parts and not parts[0].strip():
        parts.pop(0)
    while parts and not parts[-1].strip():
        parts.pop()
    return [p.strip().replace("\\|", "|") for p in parts]


def _parse_md_table(text: str) -> list[dict[str, str]]:
    """Parse a Markdown pipe table into a list of row dicts."""
    lines = [ln for ln in text.splitlines() if re.match(r"^\s*\|", ln)]
    if len(lines) < 3:          # need header + separator + at least 1 data row
        return []
    headers = _split_pipe_row(lines[0])
    rows: list[dict[str, str]] = []
    for line in lines[2:]:      # skip header (0) and separator (1)
        cells = _split_pipe_row(line)
        if len(cells) == len(headers):
            rows.append(dict(zip(headers, cells)))
    return rows


def _sev_style(val: str) -> str:
    return {
        "Critical":      "background-color: #8b0000; color: #ffffff",
        "High":          "background-color: #b84c00; color: #ffffff",
        "Medium":        "background-color: #7a5c00; color: #ffffff",
        "Low":           "background-color: #3a3a3a; color: #cccccc",
        "Informational": "background-color: #1e1e2e; color: #888888",
    }.get(val, "")


def _change_style(val: str) -> str:
    if "WORSENED"  in val: return "background-color: #8b0000; color: #ffffff"
    if "NEW"       in val: return "background-color: #003d99; color: #ffffff"
    if "IMPROVED"  in val: return "background-color: #004d1a; color: #ffffff"
    if "UNCHANGED" in val: return "background-color: #2a2a2a; color: #aaaaaa"
    return ""


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 1 — Dashboard                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def page_dashboard() -> None:
    st.header("📊 Dashboard")

    reports = sorted(_OUTPUTS.glob("*.md"))
    if not reports:
        st.info(
            "No reports yet. Go to **📤 Generate Report** to create your first one."
        )
        return

    # Render cards in rows of 3
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
                    col_b.metric(
                        "🔴 High+Critical",
                        meta["high_critical"],
                        delta_color="inverse",
                    )
                    if st.button(
                        "View Report →",
                        key=f"view_{meta['filename']}",
                        use_container_width=True,
                    ):
                        st.session_state["selected_report"] = meta["filename"]
                        st.session_state["_nav_to"] = "📄 View Report"
                        st.rerun()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 2 — Generate Report                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def page_generate() -> None:
    st.header("📤 Generate Report")
    st.caption("Upload two earnings call transcripts and let Claude compare them.")

    with st.form("pipeline_form"):
        company = st.text_input(
            "Company ticker",
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
            "🚀 Generate Report",
            use_container_width=True,
            type="primary",
        )

    if not submitted:
        return

    # ── Validation ───────────────────────────────────────────────────────────
    _period_re = re.compile(r"^\d{4}Q[1-4]$")
    errors: list[str] = []

    if not company.strip():
        errors.append("Company ticker is required.")
    if not now_period.strip():
        errors.append("Current quarter is required.")
    elif not _period_re.match(now_period.strip()):
        errors.append("Current quarter must match format YYYYQ# (e.g. 2025Q4).")
    if not prev_period.strip():
        errors.append("Prior quarter is required.")
    elif not _period_re.match(prev_period.strip()):
        errors.append("Prior quarter must match format YYYYQ# (e.g. 2025Q3).")
    if now_file is None:
        errors.append("Current quarter transcript file is required.")
    if prev_file is None:
        errors.append("Prior quarter transcript file is required.")

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

    now_ext  = Path(now_file.name).suffix.lower()   # ".txt" or ".pdf"
    prev_ext = Path(prev_file.name).suffix.lower()

    now_save_path  = company_dir / f"{now_period}_transcript{now_ext}"
    prev_save_path = company_dir / f"{prev_period}_transcript{prev_ext}"

    now_save_path.write_bytes(now_file.read())
    prev_save_path.write_bytes(prev_file.read())

    # ── Run pipeline ─────────────────────────────────────────────────────────
    try:
        from src.main import run_pipeline  # deferred import; resolves src/ path

        with st.spinner("⏳ Analysing transcripts with Claude…"):
            report_path = run_pipeline(
                company     = company,
                now_period  = now_period,
                prev_period = prev_period,
                now_path    = str(now_save_path),
                prev_path   = str(prev_save_path),
            )

        st.success("✅ Report generated successfully!")
        st.balloons()
        st.session_state["selected_report"] = os.path.basename(report_path)
        st.session_state["_nav_to"] = "📄 View Report"
        st.rerun()

    except Exception as exc:
        st.error(f"Pipeline failed: {exc}")
        st.code(traceback.format_exc())


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 3 — View Report                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def page_view() -> None:
    selected = st.session_state.get("selected_report")
    if not selected:
        st.info("Select a report from the **📊 Dashboard** first.")
        st.stop()

    report_path = _OUTPUTS / selected
    if not report_path.exists():
        st.error(f"Report file not found: `{selected}`")
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

    st.header(f"📄 {company_val} — Red Flag Report")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Company",          company_val)
    col2.metric("Current quarter",  now_val)
    col3.metric("Prior quarter",    prev_val)
    col4.metric("🔴 High+Critical", high_critical)
    st.caption(f"Generated {ts_val} · {total_changes} total changes detected")

    # ── Disclaimer ───────────────────────────────────────────────────────────
    disc_m = re.search(r"> ⚠️ \*\*Disclaimer:\*\* (.+?)(?=\n\n|\Z)", hdr, re.DOTALL)
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
    st.subheader("🚩 Red Flags")
    if rf_sec:
        rows = _parse_md_table(rf_sec)
        # Drop the "no changes" placeholder row if present
        rows = [r for r in rows if r.get("#", "—") != "—"]

        if not rows:
            st.info("No material changes detected in this report.")
        else:
            df = pd.DataFrame(rows)

            # ── Filters ──────────────────────────────────────────────────────
            all_sevs = ["Critical", "High", "Medium", "Low", "Informational"]
            all_cats = sorted(df["Category"].unique().tolist()) if "Category" in df.columns else []

            fc1, fc2 = st.columns(2)
            sel_sev = fc1.multiselect(
                "Filter by severity",
                options=all_sevs,
                default=all_sevs,
            )
            sel_cat = fc2.multiselect(
                "Filter by category",
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
        st.info("Red Flags section not found in this report.")

    st.divider()

    # ── Monitor Checklist ────────────────────────────────────────────────────
    chk_sec = sections.get("Monitor Checklist", "")
    if chk_sec:
        st.subheader("✅ Monitor Checklist")
        items = re.findall(r"- \[ \] (.+)", chk_sec)
        for i, item in enumerate(items):
            st.checkbox(item, value=False, key=f"chk_{selected}_{i}")
        st.caption("Check items manually as you verify each point in the original transcript.")

    # ── Limitations & Methodology ────────────────────────────────────────────
    lim_sec  = sections.get("Limitations", "")
    meth_sec = sections.get("Methodology", "")
    combined = "\n\n".join(s for s in [lim_sec, meth_sec] if s)
    if combined:
        with st.expander("⚠️ Limitations & Methodology", expanded=False):
            st.markdown(combined)

    st.divider()

    # ── Download button ──────────────────────────────────────────────────────
    with open(report_path, encoding="utf-8") as fh:
        raw_content = fh.read()

    st.download_button(
        label="⬇️ Download .md report",
        data=raw_content,
        file_name=selected,
        mime="text/markdown",
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Router                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

if page == "📊 Dashboard":
    page_dashboard()
elif page == "📤 Generate Report":
    page_generate()
elif page == "📄 View Report":
    page_view()
