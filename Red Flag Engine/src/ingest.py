from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Doc:
    company: str
    period: str
    text: str


def load_doc(company: str, period: str, filepath: str | Path) -> Doc:
    """Load a transcript from a PDF or TXT file and return a Doc.

    Args:
        company:  Company identifier (e.g. "AAPL").
        period:   Period label (e.g. "Q4_2024").
        filepath: Absolute or relative path to a .pdf or .txt file.

    Returns:
        Doc with the full extracted text.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is not .pdf or .txt.
        RuntimeError: If PDF extraction yields no text.
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"Transcript file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".txt":
        text = path.read_text(encoding="utf-8", errors="ignore")
        logger.debug("Loaded TXT %s (%d chars)", path.name, len(text))
        return Doc(company=company, period=period, text=text)

    if suffix == ".pdf":
        try:
            import fitz  # pymupdf
        except ImportError as exc:
            raise ImportError(
                "pymupdf is required for PDF ingestion. "
                "Run: pip install pymupdf"
            ) from exc

        pages: list[str] = []
        with fitz.open(str(path)) as pdf:
            for page in pdf:
                pages.append(page.get_text())

        text = "\n".join(pages)
        if not text.strip():
            raise RuntimeError(
                f"PDF extraction returned no text for {path}. "
                "The file may be scanned/image-based."
            )
        logger.debug("Loaded PDF %s (%d pages, %d chars)", path.name, len(pages), len(text))
        return Doc(company=company, period=period, text=text)

    raise ValueError(
        f"Unsupported file type '{suffix}' for {path}. Expected .pdf or .txt."
    )
