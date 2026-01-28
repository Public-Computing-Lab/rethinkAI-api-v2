"""
Download PDFs from Dorchester Reporter's WordPress uploads.

This script fetches PDF issues directly from the WP content paths:
https://www.dotnews.com/wp-content/uploads/YYYY/MM/

PDFs are renamed from their original format (e.g., REP-4_26web.pdf) to a
human-readable format: Dorchester_News_YYYY-MM-DD.pdf

The publication date is extracted from the PDF header text.

A sync state file tracks downloaded files to avoid re-downloading.
"""

import sys
import json
import re
import requests
from pypdf import PdfReader
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from typing import Optional
from dataclasses import dataclass, asdict
from io import BytesIO

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

import config


# Constants
DOTNEWS_SYNC_STATE_FILENAME = ".sync_state_dotnews.json"
PDF_PATTERN = re.compile(r"REP[-_]?\d+[-_]?\d*web\.pdf", re.IGNORECASE)

# Date pattern in PDF header: "Thursday, January 8, 2026"
PDF_DATE_PATTERN = re.compile(r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+" r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+" r"(\d{1,2}),?\s+(\d{4})", re.IGNORECASE)

MONTH_MAP = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12}


def extract_date_from_pdf(pdf_content: bytes) -> Optional[date]:
    """
    Extract the publication date from the PDF header.

    The Dorchester Reporter has a header like:
    "Volume 43 Issue 2 Thursday, January 8, 2026 50¢"

    Args:
        pdf_content: Raw PDF bytes

    Returns:
        date object if found, None otherwise
    """
    try:
        reader = PdfReader(BytesIO(pdf_content))
        if not reader.pages:
            return None

        # Extract text from first page only
        first_page = reader.pages[0]
        text = first_page.extract_text() or ""

        # Search for date pattern
        match = PDF_DATE_PATTERN.search(text)
        if match:
            month_name, day_str, year_str = match.groups()
            month = MONTH_MAP.get(month_name.lower())
            if month:
                return date(int(year_str), month, int(day_str))

    except Exception as e:
        print(f"    Warning: Could not extract date from PDF: {e}")

    return None


@dataclass
class SyncState:
    """Tracks downloaded files to avoid re-downloading."""

    # Maps original filename -> renamed filename
    downloaded_files: dict[str, str]
    last_sync: Optional[str] = None

    @classmethod
    def load(cls, path: Path) -> "SyncState":
        """Load sync state from file, or create new if doesn't exist."""
        if path.exists():
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                return cls(downloaded_files=data.get("downloaded_files", {}), last_sync=data.get("last_sync"))
            except (json.JSONDecodeError, KeyError):
                pass
        return cls(downloaded_files={})

    def save(self, path: Path) -> None:
        """Save sync state to file."""
        self.last_sync = datetime.now().isoformat()
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    def is_downloaded(self, original_filename: str, year: int, month: int) -> bool:
        """Check if a file has already been downloaded."""
        key = f"{year}/{month:02d}/{original_filename}"
        return key in self.downloaded_files

    def mark_downloaded(self, original_filename: str, renamed_filename: str, year: int, month: int) -> None:
        """Mark a file as downloaded."""
        key = f"{year}/{month:02d}/{original_filename}"
        self.downloaded_files[key] = renamed_filename


def generate_renamed_filename(extracted_date: Optional[date], year: int, month: int) -> str:
    """
    Generate a human-readable filename.

    Format: Dorchester_News_YYYY-MM-DD.pdf

    Args:
        extracted_date: Date extracted from PDF header (preferred)
        year: Fallback year from directory path
        month: Fallback month from directory path

    Returns:
        Renamed filename string
    """
    if extracted_date:
        return f"Dorchester_Reporter_{extracted_date.isoformat()}.pdf"

    # Fallback: use 1st of the month from directory
    fallback_date = date(year, month, 1)
    return f"Dorchester_News_{fallback_date.isoformat()}.pdf"


def list_pdfs_in_month(year: int, month: int) -> list[tuple[str, str]]:
    """
    List PDF files in a given month's WP uploads directory.

    Returns list of (filename, full_url) tuples.
    """
    url = f"{config.DOTNEWS_BASE_URL}{config.DOTNEWS_WP_UPLOADS_PATH}/{year}/{month:02d}/"
    print(f"  Checking {url}...")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"  Error fetching directory listing: {e}")
        return []

    soup = BeautifulSoup(response.content, "html.parser")
    pdfs = []

    # Look for links to PDF files matching our pattern
    for link in soup.find_all("a", href=True):
        href = link.get("href", "")
        filename = href.split("/")[-1] if "/" in href else href

        # Check if it matches our PDF pattern
        if PDF_PATTERN.match(filename):
            # Build full URL
            if href.startswith("http"):
                full_url = href
            elif href.startswith("/"):
                full_url = f"{config.DOTNEWS_BASE_URL}{href}"
            else:
                full_url = f"{url}{href}"

            pdfs.append((filename, full_url))

    # Deduplicate while preserving order
    seen = set()
    unique_pdfs = []
    for item in pdfs:
        if item[0] not in seen:
            seen.add(item[0])
            unique_pdfs.append(item)

    return unique_pdfs


def download_pdf_content(url: str) -> Optional[bytes]:
    """Download PDF and return content as bytes (for date extraction before saving)."""
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        content = response.content

        # Validate it's a PDF
        if not content[:4].startswith(b"%PDF"):
            content_type = response.headers.get("Content-Type", "").lower()
            print(f"    Error: Content is not a PDF (Content-Type: {content_type})")
            return None

        return content

    except requests.RequestException as e:
        print(f"    Error downloading: {e}")
        return None


def download_pdfs(output_dir: Optional[Path] = None, start_year: Optional[int] = None, start_month: Optional[int] = None, end_year: Optional[int] = None, end_month: Optional[int] = None) -> list[Path]:
    """
    Download PDFs from Dorchester Reporter WP uploads.

    Args:
        output_dir: Directory to save PDFs. Defaults to config.TEMP_DOWNLOAD_DIR.
        start_year: Starting year for historical download. Defaults to current year.
        start_month: Starting month for historical download. Defaults to current month.
        end_year: Ending year. Defaults to current year.
        end_month: Ending month. Defaults to current month.

    Returns:
        List of paths to downloaded PDF files.
    """
    # Set up output directory
    if output_dir is None:
        output_dir = Path(config.TEMP_DOWNLOAD_DIR)
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up date range
    today = date.today()
    if start_year is None:
        start_year = today.year
    if start_month is None:
        start_month = today.month
    if end_year is None:
        end_year = today.year
    if end_month is None:
        end_month = today.month

    # Load sync state
    sync_state_path = config.DOTNEWS_SYNC_STATE_FILENAME
    sync_state = SyncState.load(sync_state_path)

    print("Downloading Dorchester Reporter PDFs")
    print(f"  Output directory: {output_dir}")
    print(f"  Date range: {start_year}-{start_month:02d} to {end_year}-{end_month:02d}")
    print(f"  Previously downloaded: {len(sync_state.downloaded_files)} files")
    print()

    downloaded_paths = []
    current = date(start_year, start_month, 1)
    end = date(end_year, end_month, 1)

    while current <= end:
        year, month = current.year, current.month
        print(f"Processing {year}-{month:02d}...")

        pdfs = list_pdfs_in_month(year, month)
        if not pdfs:
            print(f"  No PDFs found for {year}-{month:02d}")
        else:
            print(f"  Found {len(pdfs)} PDF(s)")

            for idx, (original_filename, pdf_url) in enumerate(pdfs):
                # Check if already downloaded
                if sync_state.is_downloaded(original_filename, year, month):
                    renamed = sync_state.downloaded_files[f"{year}/{month:02d}/{original_filename}"]
                    print(f"    Skipping {original_filename} (already downloaded as {renamed})")
                    continue

                # Download PDF content to memory first
                print(f"    Downloading {original_filename}...")
                pdf_content = download_pdf_content(pdf_url)
                if pdf_content is None:
                    continue

                # Extract date from PDF header
                extracted_date = extract_date_from_pdf(pdf_content)
                if extracted_date:
                    print(f"    Extracted date from PDF: {extracted_date.isoformat()}")
                else:
                    print(f"    Warning: Could not extract date, using fallback")

                # Generate new filename
                renamed_filename = generate_renamed_filename(extracted_date, year, month)

                # Handle filename collisions
                output_path = output_dir / renamed_filename
                collision_idx = 1
                while output_path.exists():
                    base, ext = renamed_filename.rsplit(".", 1)
                    output_path = output_dir / f"{base}_{collision_idx}.{ext}"
                    collision_idx += 1

                # Save to file
                with open(output_path, "wb") as f:
                    f.write(pdf_content)

                print(f"    ✓ Saved as: {output_path.name} ({len(pdf_content) / 1024:.1f} KB)")
                sync_state.mark_downloaded(original_filename, output_path.name, year, month)
                downloaded_paths.append(output_path)

        # Move to next month
        current += relativedelta(months=1)

    # Save sync state
    sync_state.save(sync_state_path)
    print()
    print(f"Download complete. {len(downloaded_paths)} new file(s) downloaded.")
    print(f"Sync state saved to {sync_state_path}")

    return downloaded_paths


# Backwards-compatible function for existing callers
def download_latest_pdf(base_url: str = None, output_dir: Optional[Path] = None) -> Optional[Path]:  # Ignored, kept for compatibility
    """
    Download the latest PDF from the current month.

    This is a backwards-compatible wrapper around download_pdfs().

    Args:
        base_url: Ignored (kept for API compatibility)
        output_dir: Directory to save the PDF.

    Returns:
        Path to the most recently downloaded PDF, or None if none downloaded.
    """
    paths = download_pdfs(output_dir=output_dir)
    return paths[-1] if paths else None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download Dorchester Reporter PDFs from WordPress uploads")
    parser.add_argument("--start", type=str, help="Start month in YYYY-MM format (default: current month)", default=None)
    parser.add_argument("--end", type=str, help="End month in YYYY-MM format (default: current month)", default=None)
    parser.add_argument("--output-dir", type=str, help="Output directory (default: config.TEMP_DOWNLOAD_DIR)", default=None)

    args = parser.parse_args()

    # Parse start date
    start_year, start_month = None, None
    if args.start:
        try:
            start_year, start_month = map(int, args.start.split("-"))
        except ValueError:
            print(f"Error: Invalid start date format '{args.start}'. Use YYYY-MM.")
            sys.exit(1)

    # Parse end date
    end_year, end_month = None, None
    if args.end:
        try:
            end_year, end_month = map(int, args.end.split("-"))
        except ValueError:
            print(f"Error: Invalid end date format '{args.end}'. Use YYYY-MM.")
            sys.exit(1)

    # Parse output directory
    output_dir = Path(args.output_dir) if args.output_dir else None

    # Run download
    results = download_pdfs(output_dir=output_dir, start_year=start_year, start_month=start_month, end_year=end_year, end_month=end_month)

    if results:
        print(f"\nDownloaded files:")
        for path in results:
            print(f"  {path.absolute()}")
    else:
        print("\nNo new files downloaded.")
