"""Red Flag Engine — CLI entry point.

Paths are always resolved relative to the PROJECT ROOT (the directory that
contains src/, data/, and outputs/), regardless of the working directory set
in the IDE.  This means the tool works correctly whether you launch it from:
  - the project root        (python src/main.py ...)
  - the src/ subdirectory   (python main.py ...)
  - a module invocation     (python -m src.main ...)
  - an IDE run configuration with any working directory

Usage examples:
    python src/main.py --company BA   --prev 2025Q3 --now 2025Q4
    python src/main.py --company TSLA --prev 2025Q3 --now 2025Q4 --threshold 80
    python src/main.py --company BA   --prev 2025Q3 --now 2025Q4 --selfcheck

Flags:
    --company STR        Company ticker (required)
    --prev PERIOD        Prior-quarter label, e.g. 2025Q3 (required)
    --now  PERIOD        Current-quarter label, e.g. 2025Q4 (required)
    --selfcheck          Ingest+segment only — no API calls, prints chunk stats
    --data-dir PATH      Transcript root (default: <project_root>/data)
    --output-dir PATH    Report output dir (default: <project_root>/outputs)
    --threshold INT      RapidFuzz match threshold 0–100 (default: 72)
    --log-level LEVEL    DEBUG | INFO | WARNING | ERROR (default: INFO)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import Counter
from pathlib import Path

# ── Project-root anchor ────────────────────────────────────────────────────
# __file__ is  .../Red Flag Engine/src/main.py
# _PROJECT_ROOT is .../Red Flag Engine/
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# ── Load .env from project root before any import that reads env vars ──────
from dotenv import load_dotenv
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env")

# ── Ensure src/ is importable regardless of working directory ──────────────
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import anthropic

from ingest import load_doc
from segment import segment_doc
from llm_extract import extract_claims
from diff import match_claims
from report import generate_report, save_report, ReportStats


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Path helpers — always anchored to project root for relative inputs
# ---------------------------------------------------------------------------

def _resolve(path_str: str) -> Path:
    """Return an absolute Path, resolving relative paths from the project root."""
    p = Path(path_str)
    return p if p.is_absolute() else (_PROJECT_ROOT / p).resolve()


def _find_transcript(company: str, period: str, data_dir: Path) -> Path:
    """Locate <period>_transcript.pdf or .txt under data_dir/company/.

    Raises:
        FileNotFoundError: With an actionable message if no file is found.
    """
    company_dir = data_dir / company
    if not company_dir.exists():
        raise FileNotFoundError(
            f"Company directory not found: {company_dir}\n"
            f"  Expected location: {company_dir}\n"
            f"  Create it and add a transcript file named:\n"
            f"    {period}_transcript.pdf  or  {period}_transcript.txt"
        )
    for ext in (".pdf", ".txt"):
        candidate = company_dir / f"{period}_transcript{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No transcript found for {company} / {period}\n"
        f"  Looked in: {company_dir}\n"
        f"  Expected one of:\n"
        f"    {period}_transcript.pdf\n"
        f"    {period}_transcript.txt"
    )


# ---------------------------------------------------------------------------
# API key check
# ---------------------------------------------------------------------------

def _check_api_key() -> None:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        print(
            "\nERROR: ANTHROPIC_API_KEY is not set.\n"
            "\n"
            "  Fix:\n"
            "    1. Create a file named  .env  in the project root:\n"
            f"       {_PROJECT_ROOT / '.env'}\n"
            "    2. Add this line:  ANTHROPIC_API_KEY=sk-ant-...\n"
            "    3. Get a key at:   https://console.anthropic.com/\n"
            "\n"
            "  Note: A Claude Pro (claude.ai) subscription does NOT include API\n"
            "        access.  You need a separate Anthropic API account.\n",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Self-check mode (no API calls)
# ---------------------------------------------------------------------------

def run_selfcheck(
    company: str,
    prev_period: str,
    now_period: str,
    data_dir: Path,
) -> None:
    """Ingest and segment both transcripts; print diagnostic stats.

    No LLM calls are made.  Use this to verify files load correctly
    and chunking behaves as expected before spending API credits.
    """
    for period in (prev_period, now_period):
        try:
            path = _find_transcript(company, period, data_dir)
        except FileNotFoundError as exc:
            print(f"\n[SELFCHECK] ERROR: {exc}", file=sys.stderr)
            continue

        doc    = load_doc(company, period, path)
        chunks = segment_doc(doc)
        section_counts = Counter(c.section for c in chunks)

        print(f"\n{'='*60}")
        print(f"  {company}  {period}")
        print(f"{'='*60}")
        print(f"  File        : {path}")
        print(f"  Text length : {len(doc.text):,} chars")
        print(f"  Chunks      : {len(chunks)}")
        print(f"  Sections    :")
        for sec, count in sorted(section_counts.items(), key=lambda x: -x[1]):
            print(f"    {sec:<25} {count}")
        print(f"  First 2 chunks:")
        for chunk in chunks[:2]:
            preview = repr(chunk.text[:200])
            print(f"    [{chunk.chunk_id}]  section={chunk.section}  len={len(chunk.text)}")
            print(f"      {preview}")

    print()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python src/main.py",
        description="Red Flag Engine — earnings call transcript change detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python src/main.py --company BA   --prev 2025Q3 --now 2025Q4\n"
            "  python src/main.py --company TSLA --prev 2025Q3 --now 2025Q4 --threshold 80\n"
            "  python src/main.py --company BA   --prev 2025Q3 --now 2025Q4 --selfcheck\n"
        ),
    )
    parser.add_argument(
        "--company", required=True,
        help="Company ticker, e.g. BA.",
    )
    parser.add_argument(
        "--prev", required=True, metavar="PERIOD",
        help="Prior-quarter period label, e.g. 2025Q3.",
    )
    parser.add_argument(
        "--now", required=True, metavar="PERIOD",
        help="Current-quarter period label, e.g. 2025Q4.",
    )
    parser.add_argument(
        "--selfcheck", action="store_true",
        help=(
            "Ingest and segment both transcripts without making any API calls. "
            "Prints text length, chunk count, section distribution, and previews "
            "of the first 2 chunks for each period.  Use this to verify your "
            "data files are correctly placed and formatted."
        ),
    )
    parser.add_argument(
        "--data-dir", default="data", metavar="PATH",
        help=(
            "Root directory containing company transcript folders. "
            "Relative paths are anchored to the project root. "
            "(default: data/)"
        ),
    )
    parser.add_argument(
        "--output-dir", default="outputs", metavar="PATH",
        help=(
            "Directory to write Markdown reports into. "
            "Relative paths are anchored to the project root. "
            "(default: outputs/)"
        ),
    )
    parser.add_argument(
        "--threshold", type=int, default=72, metavar="INT",
        help="RapidFuzz token_set_ratio match threshold 0–100 (default: 72).",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Streamlit-callable pipeline (explicit file paths, no _find_transcript)
# ---------------------------------------------------------------------------

def run_pipeline(
    company:     str,
    now_period:  str,
    prev_period: str,
    now_path:    str,
    prev_path:   str,
) -> str:
    """Run the full Red Flag Engine pipeline with explicit file paths.

    Unlike ``run()``, this function accepts absolute file paths directly
    (useful when the caller — e.g. a Streamlit UI — already has the file
    on disk) rather than deriving them from a company / period / data-dir
    triple.

    Args:
        company:     Company ticker (e.g. "BA").
        now_period:  Label for the current quarter (e.g. "2025Q4").
        prev_period: Label for the prior quarter (e.g. "2025Q3").
        now_path:    Absolute path to the current-quarter transcript.
        prev_path:   Absolute path to the prior-quarter transcript.

    Returns:
        Absolute path to the generated Markdown report as a string.

    Raises:
        RuntimeError: If ingestion or segmentation yields empty results.
        Any exception from the underlying pipeline steps is propagated.
    """
    if not logging.root.handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    log = logging.getLogger(__name__)

    output_dir = _PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    now_p  = Path(now_path)
    prev_p = Path(prev_path)

    log.info("=== run_pipeline ===")
    log.info("Company       : %s", company)
    log.info("Now period    : %s  (%s)", now_period,  now_p)
    log.info("Prior period  : %s  (%s)", prev_period, prev_p)

    # 1. Ingest
    doc_now  = load_doc(company, now_period,  now_p)
    doc_prev = load_doc(company, prev_period, prev_p)

    if not doc_now.text.strip():
        raise RuntimeError(f"Current transcript is empty after ingestion: {now_p}")
    if not doc_prev.text.strip():
        raise RuntimeError(f"Prior transcript is empty after ingestion: {prev_p}")

    # 2. Segment
    chunks_now  = segment_doc(doc_now)
    chunks_prev = segment_doc(doc_prev)

    if not chunks_now:
        raise RuntimeError(f"Segmentation produced 0 chunks for: {now_p}")
    if not chunks_prev:
        raise RuntimeError(f"Segmentation produced 0 chunks for: {prev_p}")

    log.info("Chunks: now=%d  prev=%d", len(chunks_now), len(chunks_prev))

    # 3. Extract claims
    client = anthropic.Anthropic()

    claims_now  = extract_claims(chunks_now,  client)
    claims_prev = extract_claims(chunks_prev, client)

    log.info("Claims: now=%d  prev=%d", len(claims_now), len(claims_prev))

    # 4. Diff
    changes = match_claims(claims_now, claims_prev)

    # 5. Report
    stats = ReportStats(
        n_chunks_now  = len(chunks_now),
        n_chunks_prev = len(chunks_prev),
        n_claims_now  = len(claims_now),
        n_claims_prev = len(claims_prev),
        n_matched     = sum(1 for c in changes if c.change_type.value != "new"),
        n_new         = sum(1 for c in changes if c.change_type.value == "new"),
        n_soft        = sum(1 for c in changes if c.match_quality == "soft"),
    )
    report_md = generate_report(company, now_period, prev_period, changes, stats)
    out_path  = save_report(report_md, company, now_period, prev_period, output_dir)

    return str(out_path.resolve())


# ---------------------------------------------------------------------------
# Full pipeline (CLI-oriented: derives paths from company/period/data_dir)
# ---------------------------------------------------------------------------

def run(
    company: str,
    prev_period: str,
    now_period: str,
    data_dir: Path,
    output_dir: Path,
    threshold: int,
) -> Path:
    """Execute the full pipeline and return the path to the written report."""
    log = logging.getLogger(__name__)

    prev_path = _find_transcript(company, prev_period, data_dir)
    now_path  = _find_transcript(company, now_period,  data_dir)

    log.info("=== Red Flag Engine ===")
    log.info("Project root  : %s", _PROJECT_ROOT)
    log.info("Company       : %s", company)
    log.info("Prior quarter : %s  (%s)", prev_period, prev_path)
    log.info("Now quarter   : %s  (%s)", now_period, now_path)
    log.info("Threshold     : %d", threshold)
    log.info("Output dir    : %s", output_dir)

    # 1. Ingest
    log.info("[1/5] Ingesting transcripts…")
    doc_prev = load_doc(company, prev_period, prev_path)
    doc_now  = load_doc(company, now_period,  now_path)

    if not doc_prev.text.strip():
        raise RuntimeError(f"Prior transcript is empty after ingestion: {prev_path}")
    if not doc_now.text.strip():
        raise RuntimeError(f"Current transcript is empty after ingestion: {now_path}")

    # 2. Segment
    log.info("[2/5] Segmenting transcripts…")
    chunks_prev = segment_doc(doc_prev)
    chunks_now  = segment_doc(doc_now)

    if not chunks_prev:
        raise RuntimeError(f"Segmentation produced 0 chunks for: {prev_path}")
    if not chunks_now:
        raise RuntimeError(f"Segmentation produced 0 chunks for: {now_path}")

    log.info("  Prior: %d chunks  |  Now: %d chunks", len(chunks_prev), len(chunks_now))

    # 3. Extract claims (both quarters)
    client = anthropic.Anthropic()

    log.info("[3/5] Extracting claims — prior quarter (%d chunks)…", len(chunks_prev))
    claims_prev = extract_claims(chunks_prev, client)

    log.info("[3/5] Extracting claims — current quarter (%d chunks)…", len(chunks_now))
    claims_now = extract_claims(chunks_now, client)

    log.info("  Claims: prior=%d  now=%d", len(claims_prev), len(claims_now))

    # 4. Diff
    log.info("[4/5] Running change detection (threshold=%d)…", threshold)
    changes = match_claims(claims_now, claims_prev, threshold=threshold)

    # 5. Report
    log.info("[5/5] Generating Markdown report…")
    stats = ReportStats(
        n_chunks_now  = len(chunks_now),
        n_chunks_prev = len(chunks_prev),
        n_claims_now  = len(claims_now),
        n_claims_prev = len(claims_prev),
        n_matched     = sum(1 for c in changes if c.change_type.value != "new"),
        n_new         = sum(1 for c in changes if c.change_type.value == "new"),
        n_soft        = sum(1 for c in changes if c.match_quality == "soft"),
    )
    report_md = generate_report(company, now_period, prev_period, changes, stats)
    out_path  = save_report(report_md, company, now_period, prev_period, output_dir)

    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    _configure_logging(args.log_level)

    data_dir   = _resolve(args.data_dir)
    output_dir = _resolve(args.output_dir)

    # ── Self-check: no API needed ──────────────────────────────────────────
    if args.selfcheck:
        run_selfcheck(
            company=args.company,
            prev_period=args.prev,
            now_period=args.now,
            data_dir=data_dir,
        )
        return

    # ── Full pipeline: API required ────────────────────────────────────────
    _check_api_key()

    try:
        out_path = run(
            company=args.company,
            prev_period=args.prev,
            now_period=args.now,
            data_dir=data_dir,
            output_dir=output_dir,
            threshold=args.threshold,
        )
    except FileNotFoundError as exc:
        print(f"\nERROR (file not found):\n{exc}\n", file=sys.stderr)
        sys.exit(1)
    except (ValueError, RuntimeError) as exc:
        print(f"\nERROR: {exc}\n", file=sys.stderr)
        sys.exit(1)

    print(f"\nDone. Report written to:\n  {out_path}\n")


if __name__ == "__main__":
    main()
