# Red Flag Engine

A local-only CLI that compares two quarters of earnings call transcripts, extracts
auditable claims, detects quarter-over-quarter changes, and writes a Markdown report.
Every claim in the output is grounded in a verbatim evidence quote and a chunk ID so
analysts can trace any finding back to the original transcript.

---

## Setup

**Requirements:** Python 3.11+

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create your environment file
cp .env.example .env          # or create .env manually

# 3. Add your Anthropic API key to .env
#    ANTHROPIC_API_KEY=sk-ant-...
```

---

## Data layout

Store transcripts under `data/<COMPANY>/` using the naming convention:

```
data/
  AAPL/
    Q3_2024_transcript.txt
    Q4_2024_transcript.txt
  NVDA/
    Q3_2024_transcript.pdf
    Q4_2024_transcript.pdf
```

Both `.pdf` and `.txt` files are supported.
The period label in the filename (`Q3_2024`) is used automatically in the report.

---

## Usage

```bash
python src/main.py \
    --company AAPL \
    --prev    data/AAPL/Q3_2024_transcript.txt \
    --now     data/AAPL/Q4_2024_transcript.txt
```

The report is written to `outputs/AAPL_Q4_2024_vs_Q3_2024.md` and the path is
printed on completion.

### All flags

| Flag | Default | Description |
|------|---------|-------------|
| `--company` | *(required)* | Company identifier used in filenames and report headers |
| `--prev` | *(required)* | Path to the prior-quarter transcript |
| `--now` | *(required)* | Path to the current-quarter transcript |
| `--threshold` | `72` | RapidFuzz `token_set_ratio` match threshold (0–100) |
| `--output-dir` | `outputs/` | Directory for Markdown report output |
| `--log-level` | `INFO` | Logging verbosity: `DEBUG` / `INFO` / `WARNING` / `ERROR` |

---

## Report structure

Each report contains:

1. **Header table** — company, quarters, generation timestamp, change counts
2. **Executive Summary** — top 7 changes with severity labels
3. **Red Flags Table** — top 10 changes with evidence quotes and chunk IDs
4. **Monitor Checklist** — five analyst action items (guidance, demand, margins, liquidity, regulatory)
5. **Methodology Note** — brief description of chunking and extraction approach

---

## Limitations

- **Triage tool only.** This is not a trading signal. Do not use this tool to make
  investment decisions.
- **LLM extraction may produce false positives or miss subtle language.** All output
  should be reviewed by a qualified analyst before any action is taken.
- **Claims are bounded by transcript quality and available context.** Poor-quality
  PDFs, incomplete transcripts, or heavily redacted documents will reduce accuracy.
- **No financial advice is provided or implied.** The authors accept no liability for
  decisions made based on output from this tool.
- **MVP scope is earnings call transcripts only.** Filings (10-Q/10-K MD&A) are not
  supported in this version.

---

## Project structure

```
redflag-engine/
  data/               # Local transcripts — not committed to version control
  src/
    ingest.py         # PDF/TXT → Doc
    segment.py        # Doc → list[Chunk]
    llm_extract.py    # list[Chunk] → list[Claim]  (Claude API)
    diff.py           # list[Claim] × 2 → list[Change]
    report.py         # list[Change] → Markdown
    main.py           # CLI entry point
  outputs/            # Generated reports
  requirements.txt
  .env                # API key — never committed
  README.md
```

---

## How to read a Red Flag Report

**Severity scale:** 1 (Informational) → 5 (Critical). Focus on severity 4–5 first.

**Change types:**
- `NEW` — no matching claim found in the prior quarter.
- `WORSENED` — same topic found in prior quarter, sentiment shifted negative.
- `IMPROVED` — same topic found in prior quarter, sentiment shifted positive.
- `UNCHANGED` — same topic, no meaningful sentiment change.

**⁽ᴬ⁾ marker:** The claim originates from an analyst question, not a management statement. Analyst questions may reflect hypotheticals or concerns that management did not endorse. Apply lower weight.

**⁽ˢ⁾ marker:** Matched via relaxed similarity (soft match — same category, first 60 characters). The link between current and prior quarter may be weaker. Verify manually against the original transcript before acting.

**Coverage line:** The `📊 Coverage` line in the report header shows how many chunks were analysed, how many claims were extracted per quarter, and how many were matched vs. new. Low claim counts relative to chunk counts may indicate extraction failures (check `--log-level DEBUG`).

**This report is a triage aid.** It surfaces candidates for analyst review. It is not a trading signal and must not be used as a basis for investment decisions. No financial advice is provided or implied.

---

## Streamlit Web Interface

A browser-based UI is included in `streamlit_app.py`. It provides three pages:

- **Dashboard** — cards for every report in `outputs/`, showing change counts and severity at a glance.
- **Generate Report** — upload two transcripts, enter ticker + quarters, and run the full pipeline in-browser.
- **View Report** — interactive table with severity/category filters, colour-coded cells, and a Markdown download button.

### Running locally

```bash
# 1. Install all dependencies (including streamlit and pandas)
pip install -r requirements.txt

# 2. Add your API key to .streamlit/secrets.toml
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
#    Then edit .streamlit/secrets.toml and set ANTHROPIC_API_KEY=sk-ant-...

# 3. Launch
streamlit run streamlit_app.py
# → opens http://localhost:8501
```

### Deploying to Streamlit Community Cloud (shareable URL)

1. Push the repo to GitHub. Confirm `.env` and `.streamlit/secrets.toml` are in `.gitignore` — **never commit real secrets**.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → select your repo → set main file to `streamlit_app.py`.
3. In **Advanced settings → Secrets**, paste:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-your-key-here"
   ```
4. Click **Deploy**. You receive a public URL such as `https://redflag-engine.streamlit.app`.
5. Share that URL — no login required to view.

> **Note:** Transcript files uploaded via the web UI are saved to `data/<COMPANY>/` within the deployed app's ephemeral file system. They do not persist across app restarts on Community Cloud. The generated Markdown reports in `outputs/` are similarly ephemeral. For persistent storage, download the `.md` report using the in-page download button.
