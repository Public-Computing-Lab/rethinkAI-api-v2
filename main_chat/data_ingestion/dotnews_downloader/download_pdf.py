"""
Download PDFs from Dorchester Reporter's WordPress uploads.
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
from dataclasses import dataclass, field
from io import BytesIO

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

import config
from main_chat.data_ingestion.utils.log_util import log_debug, log_info, log_error, log_warning, log_success

# PDF filename pattern
PDF_PATTERN = re.compile(r"REP[-_]?\d+[-_]?\d*web\.pdf", re.IGNORECASE)
PDF_DATE_PATTERN = re.compile(r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+" r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+" r"(\d{1,2}),?\s+(\d{4})", re.IGNORECASE)
PDF_HREF_PATTERN = re.compile(r'href=["\']([^"\']*REP[-_]?\d+[-_]?\d*web\.pdf)["\']', re.IGNORECASE)

MONTH_MAP = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12}

DOTNEWS_BASE_URL = "https://www.dotnews.com"
DOTNEWS_WP_UPLOADS_PATH = "/wp-content/uploads"


def extract_date_from_pdf(pdf_content: bytes) -> Optional[date]:
    """Extract the publication date from the PDF header."""
    try:
        reader = PdfReader(BytesIO(pdf_content))
        if not reader.pages:
            return None
        text = reader.pages[0].extract_text() or ""
        match = PDF_DATE_PATTERN.search(text)
        if match:
            month_name, day_str, year_str = match.groups()
            month = MONTH_MAP.get(month_name.lower())
            if month:
                return date(int(year_str), month, int(day_str))
    except Exception as e:
        log_warning(f"Could not extract date from PDF: {e}")
    return None


@dataclass
class SyncState:
    """
    Tracks downloaded files by original filename to avoid re-downloading.

    Format:
    {
        "downloaded_files": {
            "REP-1_8web.pdf": {
                "renamed": "Dorchester_Reporter_2026-01-08.pdf",
                "downloaded_at": "2026-01-08T10:00:00",
                "processed_at": null,  # Set by main_daily_ingestion after processing
                "events": null
            }
        },
        "last_sync": "2026-01-08T10:00:00"
    }
    """

    downloaded_files: dict = field(default_factory=dict)
    last_sync: Optional[str] = None

    @classmethod
    def load(cls, path: Path) -> "SyncState":
        if path.exists():
            try:
                with open(path, "r") as f:
                    data = json.load(f)

                # Handle legacy format (keyed by renamed filename)
                # Migrate if we detect old format
                raw_files = data.get("downloaded_files", data)
                if raw_files and not data.get("downloaded_files"):
                    # Old format: entire file is {renamed_filename: {...}}
                    # We can't recover original filenames, so start fresh
                    # but keep track of renamed files to avoid reprocessing
                    log_info("Migrating legacy sync state format...")
                    return cls(downloaded_files={}, last_sync=data.get("last_sync"))

                return cls(downloaded_files=data.get("downloaded_files", {}), last_sync=data.get("last_sync"))
            except (json.JSONDecodeError, KeyError, IOError):
                pass
        return cls()

    def save(self, path: Path) -> None:
        self.last_sync = datetime.now().isoformat()
        data = {"downloaded_files": self.downloaded_files, "last_sync": self.last_sync}
        temp_path = path.with_suffix(".tmp")
        try:
            with open(temp_path, "w") as f:
                json.dump(data, f, indent=2)
            temp_path.replace(path)
        except Exception:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            if temp_path.exists():
                temp_path.unlink()

    def is_downloaded(self, original_filename: str) -> bool:
        """Check if file was already downloaded by its original filename."""
        return original_filename in self.downloaded_files

    def mark_downloaded(self, original_filename: str, renamed_filename: str) -> None:
        """Record a downloaded file. Processing info added later by main_daily_ingestion."""
        self.downloaded_files[original_filename] = {"renamed": renamed_filename, "downloaded_at": datetime.now().isoformat(), "processed_at": None, "events": None}

    def mark_processed(self, original_filename: str, events: int) -> None:
        """Mark a file as processed (called by main_daily_ingestion)."""
        if original_filename in self.downloaded_files:
            self.downloaded_files[original_filename]["processed_at"] = datetime.now().isoformat()
            self.downloaded_files[original_filename]["events"] = events

    def is_processed(self, original_filename: str) -> bool:
        """Check if file has been processed."""
        entry = self.downloaded_files.get(original_filename)
        return entry is not None and entry.get("processed_at") is not None

    def get_renamed_filename(self, original_filename: str) -> Optional[str]:
        """Get the renamed filename for an original filename."""
        entry = self.downloaded_files.get(original_filename)
        return entry.get("renamed") if entry else None

    def get_unprocessed_files(self) -> list[tuple[str, str]]:
        """Get list of (original_filename, renamed_filename) for unprocessed files."""
        return [(orig, entry["renamed"]) for orig, entry in self.downloaded_files.items() if entry.get("processed_at") is None]


def generate_renamed_filename(extracted_date: Optional[date], year: int, month: int) -> str:
    if extracted_date:
        return f"Dorchester_Reporter_{extracted_date.isoformat()}.pdf"
    fallback_date = date(year, month, 1)
    return f"Dorchester_Reporter_{fallback_date.isoformat()}.pdf"


def list_pdfs_in_month(year: int, month: int) -> list[tuple[str, str]]:
    """List PDF files in a given month's WP uploads directory."""
    url = f"{DOTNEWS_BASE_URL}{DOTNEWS_WP_UPLOADS_PATH}/{year}/{month:02d}/"
    log_success(f"Downloading from {url}...")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        log_error(f"Error fetching directory listing: {e}")
        return []

    pdfs = []
    matches = PDF_HREF_PATTERN.findall(response.text)

    if matches:
        for href in matches:
            filename = href.split("/")[-1] if "/" in href else href
            if href.startswith("http"):
                full_url = href
            elif href.startswith("/"):
                full_url = f"{DOTNEWS_BASE_URL}{href}"
            else:
                full_url = f"{url}{href}"
            pdfs.append((filename, full_url))
    else:
        soup = BeautifulSoup(response.content, "html.parser")
        for link in soup.find_all("a", href=True):
            href = link.get("href", "")
            filename = href.split("/")[-1] if "/" in href else href
            if PDF_PATTERN.match(filename):
                if href.startswith("http"):
                    full_url = href
                elif href.startswith("/"):
                    full_url = f"{DOTNEWS_BASE_URL}{href}"
                else:
                    full_url = f"{url}{href}"
                pdfs.append((filename, full_url))

    seen = set()
    unique_pdfs = []
    for item in pdfs:
        if item[0] not in seen:
            seen.add(item[0])
            unique_pdfs.append(item)
    return unique_pdfs


def download_pdf_content(url: str) -> Optional[bytes]:
    """Download PDF and return content as bytes."""
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        content = response.content
        if not content[:4].startswith(b"%PDF"):
            content_type = response.headers.get("Content-Type", "").lower()
            log_error(f"Content is not a PDF (Content-Type: {content_type})")
            return None
        return content
    except requests.RequestException as e:
        log_error(f"Error downloading: {e}")
        return None


def download_pdfs(output_dir: Optional[Path] = None, start_year: Optional[int] = None, start_month: Optional[int] = None, end_year: Optional[int] = None, end_month: Optional[int] = None) -> list[Path]:
    """
    Download PDFs from Dorchester Reporter WP uploads.

    Sync state tracks original filenames (e.g., "REP-1_8web.pdf") to avoid
    re-downloading. The check happens BEFORE download - we only fetch files
    whose original filename is not already in the sync state.
    """
    if output_dir is None:
        output_dir = config.DATA_DOWNLOAD_DIR / "dotnews"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    today = date.today()
    start_year = start_year or today.year
    start_month = start_month or today.month
    end_year = end_year or today.year
    end_month = end_month or today.month

    sync_state_path = config.DOTNEWS_SYNC_STATE_FILENAME
    sync_state = SyncState.load(sync_state_path)

    log_debug(f"Output directory: {output_dir}")
    log_debug(f"Date range: {start_year}-{start_month:02d} to {end_year}-{end_month:02d}")
    log_debug(f"Previously downloaded: {len(sync_state.downloaded_files)} files")

    downloaded_paths = []
    current = date(start_year, start_month, 1)
    end = date(end_year, end_month, 1)

    while current <= end:
        year, month = current.year, current.month
        log_debug(f"Processing {year}-{month:02d}...")

        pdfs = list_pdfs_in_month(year, month)
        if not pdfs:
            log_debug(f"  No PDFs found for {year}-{month:02d}")
        else:
            log_debug(f"  Found {len(pdfs)} PDF(s)")

            for original_filename, pdf_url in pdfs:
                # Check sync state BEFORE downloading - skip if we've seen this filename
                if sync_state.is_downloaded(original_filename):
                    log_debug(f"  Skipping {original_filename} (already downloaded)")
                    continue

                # Only download files we haven't seen before
                log_debug(f"  Downloading {original_filename}...")
                pdf_content = download_pdf_content(pdf_url)
                if pdf_content is None:
                    continue

                # Extract date from PDF content for human-readable filename
                extracted_date = extract_date_from_pdf(pdf_content)
                if extracted_date:
                    log_debug(f"  Extracted date: {extracted_date.isoformat()}")
                else:
                    log_debug("  Could not extract date, using fallback")

                renamed_filename = generate_renamed_filename(extracted_date, year, month)
                output_path = output_dir / renamed_filename

                with open(output_path, "wb") as f:
                    f.write(pdf_content)

                log_debug(f"  Saved as: {output_path.name} ({len(pdf_content) / 1024:.1f} KB)")

                # Record original filename -> renamed filename mapping
                sync_state.mark_downloaded(original_filename, renamed_filename)
                sync_state.save(sync_state_path)
                downloaded_paths.append(output_path)

        current += relativedelta(months=1)

    return downloaded_paths


if __name__ == "__main__":
    import argparse
    from main_chat.data_ingestion.utils.log_util import set_verbosity, Verbosity

    parser = argparse.ArgumentParser(description="Download Dorchester Reporter PDFs")
    parser.add_argument("--start", type=str, help="Start month YYYY-MM", default=None)
    parser.add_argument("--end", type=str, help="End month YYYY-MM", default=None)
    parser.add_argument("--output-dir", type=str, help="Output directory", default=None)
    parser.add_argument("-q", "--quiet", action="store_true", help="Minimal output")
    args = parser.parse_args()

    set_verbosity(Verbosity.NORMAL if args.quiet else Verbosity.VERBOSE)

    start_year, start_month = None, None
    if args.start:
        try:
            start_year, start_month = map(int, args.start.split("-"))
        except ValueError:
            log_error(f"Error: Invalid start date format '{args.start}'. Use YYYY-MM.")
            sys.exit(1)

    end_year, end_month = None, None
    if args.end:
        try:
            end_year, end_month = map(int, args.end.split("-"))
        except ValueError:
            log_error(f"Error: Invalid end date format '{args.end}'. Use YYYY-MM.")
            sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else None
    results = download_pdfs(output_dir=output_dir, start_year=start_year, start_month=start_month, end_year=end_year, end_month=end_month)

    if results:
        log_success(f"Downloaded {len(results)} new files.")
    else:
        log_success("No new files downloaded.")
