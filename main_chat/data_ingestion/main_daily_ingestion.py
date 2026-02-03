import sys
from pathlib import Path
from datetime import datetime
import json
import argparse
import concurrent.futures

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import config

# Import shared logging FIRST, before other modules
from main_chat.data_ingestion.utils.log_util import get_verbosity, set_verbosity, Verbosity, log, log_error, log_info, log_debug, log_success, log_warning, log_msg

# Import ingestion modules (they will use log_util)
from google_drive_to_vectordb import sync_google_drive_to_vectordb
from email_to_calendar_sql import sync_email_newsletters_to_sql, AuthenticationRequiredError
from boston_data_sync.boston_data_sync import BostonDataSyncer
from dotnews_downloader import download_pdfs, SyncState as DotnewsSyncState
from google_drive_to_vectordb import process_newsletter_pdf, insert_events_to_db


def sync_dotnews_newsletters() -> dict:
    """Download and process the latest newsletter from dotnews.com."""
    stats = {"pdfs_downloaded": 0, "pdfs_processed": 0, "events_extracted": 0, "chunks_added": 0, "errors": []}

    try:
        dotnews_dir = config.DATA_DOWNLOAD_DIR / "dotnews"
        dotnews_dir.mkdir(parents=True, exist_ok=True)

        # Download new PDFs (uses SyncState internally to skip already-downloaded)
        pdf_paths = download_pdfs(output_dir=dotnews_dir)
        stats["pdfs_downloaded"] = len(pdf_paths)

        # Load sync state to check what needs processing
        sync_state_path = config.DOTNEWS_SYNC_STATE_FILENAME
        sync_state = DotnewsSyncState.load(sync_state_path)

        # Get all unprocessed files (includes newly downloaded + any previous failures)
        unprocessed = sync_state.get_unprocessed_files()

        if not unprocessed:
            log_info("No new PDFs to process")
            return stats

        log_info(f"Processing {len(unprocessed)} PDF(s)")

        for original_filename, renamed_filename in unprocessed:
            pdf_path = dotnews_dir / renamed_filename

            if not pdf_path.exists():
                log_warning(f"File not found: {renamed_filename}")
                continue

            log_debug(f"  Processing: {renamed_filename}")
            file_metadata = {"name": renamed_filename, "id": f"dotnews_{renamed_filename}", "modifiedTime": datetime.fromtimestamp(pdf_path.stat().st_mtime).isoformat() + "Z"}

            result = process_newsletter_pdf(pdf_path, file_metadata)

            events_count = 0
            if result.get("events"):
                events_inserted = insert_events_to_db(result["events"])
                events_count = len(result["events"])
                stats["events_extracted"] += events_count
                log_debug(f"  ✔  Inserted {events_inserted} events")

            stats["chunks_added"] += len(result.get("documents", []))
            stats["pdfs_processed"] += 1

            # Mark as processed in sync state
            sync_state.mark_processed(original_filename, events_count)
            sync_state.save(sync_state_path)

    except Exception as e:
        error_msg = f"Error syncing dotnews: {e}"
        log_error(error_msg)
        stats["errors"].append(error_msg)

    return stats


def safe_email_sync(interactive: bool = False) -> dict:
    """Wrapper for email sync with proper error handling."""
    try:
        email_stats = sync_email_newsletters_to_sql(interactive=interactive)
        if email_stats.get("auth_required"):
            log_error("GMAIL AUTHENTICATION REQUIRED")
            log_info(f"Visit: {email_stats.get('auth_url')}")
        return email_stats
    except AuthenticationRequiredError as e:
        log_error("Gmail authentication required!")
        log_info(f"Visit: {e.auth_url}")
        return {"emails_processed": 0, "events_extracted": 0, "events_inserted": 0, "errors": [f"Authentication required: {e.auth_url}"], "auth_required": True, "auth_url": e.auth_url}
    except Exception as e:
        log_error(f"Email sync failed: {e}")
        return {"emails_processed": 0, "events_extracted": 0, "events_inserted": 0, "errors": [str(e)]}


def safe_boston_sync() -> dict:
    """Wrapper for Boston data sync with proper error handling."""
    try:
        syncer = BostonDataSyncer()
        return syncer.sync_all()
    except Exception as e:
        log_error(f"Boston data sync failed: {e}")
        return {"datasets_synced": 0, "total_records": 0, "datasets": [{"dataset": "unknown", "errors": [str(e)]}]}


def safe_drive_sync() -> dict:
    """Wrapper for Google Drive sync with proper error handling."""
    try:
        return sync_google_drive_to_vectordb()
    except Exception as e:
        log_error(f"Google Drive sync failed: {e}")
        return {"files_processed": 0, "chunks_added": 0, "errors": [str(e)]}


def log_run_summary(drive_stats, email_stats, boston_stats=None, dotnews_stats=None):
    """Log summary of the ingestion run to a JSONL file."""
    log_file = _PROJECT_ROOT / "logs/ingestion_log.jsonl"
    summary = {
        "timestamp": datetime.now().isoformat(),
        "google_drive": drive_stats,
        "email_newsletters": email_stats,
    }
    if boston_stats:
        summary["boston_open_data"] = boston_stats
    if dotnews_stats:
        summary["dotnews"] = dotnews_stats

    errors = len(drive_stats.get("errors", [])) + len(email_stats.get("errors", [])) + (sum(len(d.get("errors", [])) for d in boston_stats.get("datasets", [])) if boston_stats else 0) + (len(dotnews_stats.get("errors", [])) if dotnews_stats else 0)
    summary["success"] = errors == 0

    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")
    except Exception as e:
        log_error(f"Could not write to log file: {e}")


def print_banner(title: str):
    width = 80
    log("\n" + "╔" + "=" * (width - 2) + "╗")
    padding = (width - len(title) - 2) // 2
    log("║" + " " * padding + title + " " * (width - padding - len(title) - 2) + "║")
    log("╚" + "=" * (width - 2) + "╝\n")


def print_final_summary(drive_stats, email_stats, boston_stats=None, dotnews_stats=None):
    """Print final summary - always shown."""
    gdrive_errors = len(drive_stats.get("errors", []))
    gmail_errors = len(email_stats.get("errors", []))
    bos311_errors = sum(len(d.get("errors", [])) for d in boston_stats.get("datasets", [])) if boston_stats else 0
    dotnews_errors = len(dotnews_stats.get("errors", [])) if dotnews_stats else 0
    total_errors = gdrive_errors + gmail_errors + bos311_errors + dotnews_errors

    # Always show final summary (force=True equivalent via direct print)
    log_msg("\n" + "╔" + "=" * 78 + "╗")
    log_msg("║" + " " * 30 + " SYNC SUMMARY" + " " * 35 + "║")
    log_msg("╠" + "=" * 78 + "╣")

    def stat_line(label, value):
        return f"  »  {label:<40} {str(value):>6}"

    log_msg(stat_line("Dotnews PDFs Processed:", dotnews_stats.get("pdfs_processed", 0) if dotnews_stats else 0))
    log_msg(stat_line("Dotnews Events Extracted:", dotnews_stats.get("events_extracted", 0) if dotnews_stats else 0))
    log_msg(stat_line("Google Drive Files Processed:", drive_stats.get("files_processed", 0)))
    log_msg(stat_line("Vector DB Chunks Added:", drive_stats.get("chunks_added", 0)))
    log_msg(stat_line("Emails Processed:", email_stats.get("emails_processed", 0)))
    log_msg(stat_line("Email Events (SQL):", email_stats.get("events_inserted", 0)))
    if boston_stats:
        log_msg(stat_line("Boston Datasets Synced:", boston_stats.get("datasets_synced", 0)))
        log_msg(stat_line("Boston Records Synced:", boston_stats.get("total_records", 0)))
    log_msg(stat_line("Total Errors:", total_errors))
    log_msg("╚" + "=" * 78 + "╝\n")

    if total_errors:
        log_error(f"Daily ingestion completed with {total_errors} error(s).\n")
        log_error("Error details:")
        for error in drive_stats.get("errors", []):
            log_error(f"[Google Drive] {error}")
        for error in email_stats.get("errors", []):
            log_error(f"[Gmail] {error}")
        if boston_stats:
            for dataset in boston_stats.get("datasets", []):
                for error in dataset.get("errors", []):
                    log_error(f"[Boston 311: {dataset.get('dataset', 'unknown')}] {error}")
        if dotnews_stats:
            for error in dotnews_stats.get("errors", []):
                log_error(f"[Dotnews] {error}")


def run_sequential():
    log_info("Running in SERIAL mode")

    # """Run all phases sequentially with full logging."""
    log("\n" + "=" * 60)
    log("▶ STARTING: Dotnews Newsletter Download & Processing")
    log("=" * 60)
    dotnews_stats = sync_dotnews_newsletters()
    log_success(f"COMPLETED: Dotnews ({dotnews_stats.get('pdfs_processed', 0)} PDFs)")

    log("\n" + "=" * 60)
    log("▶ STARTING: Google Drive → Vector DB")
    log("=" * 60)
    drive_stats = safe_drive_sync()
    log_success(f"COMPLETED: Google Drive sync({drive_stats.get('files_processed', 0)} files)")

    log("\n" + "=" * 60)
    log("▶ STARTING: Email Newsletter → Calendar SQL")
    log("=" * 60)
    email_stats = safe_email_sync(interactive=False)
    log_success(f"COMPLETED: Gmail sync ({email_stats.get('emails_processed', 0)} emails)")

    log("\n" + "=" * 60)
    log("▶ STARTING: Boston Open Data → MySQL")
    log("=" * 60)
    boston_stats = safe_boston_sync()
    log_success(f"COMPLETED: Boston 311 Data sync ({boston_stats.get('total_records', 0)} records)")

    return drive_stats, email_stats, boston_stats, dotnews_stats


def run_parallel():
    """Run phases in parallel with minimal logging."""
    log_info("Running in PARALLEL mode")

    # Dotnews first (quick)
    log("▶ Starting: Dotnews")
    dotnews_stats = sync_dotnews_newsletters()
    # errors = len(dotnews_stats.get("errors", []))
    # if errors == 0:
    #     log_success(f"Dotnews: {dotnews_stats.get('pdfs_processed', 0)} PDFs, " f"{dotnews_stats.get('events_extracted', 0)} events")
    # else:
    #     log_error(f"Dotnews dowload/processing error.")

    start_time = datetime.now()
    log("\n▶ Starting parallel: Google Drive, Email, Boston Data...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_drive = executor.submit(safe_drive_sync)
        future_email = executor.submit(safe_email_sync, False)
        future_boston = executor.submit(safe_boston_sync)

        drive_stats = future_drive.result()
        email_stats = future_email.result()
        boston_stats = future_boston.result()

    elapsed = (datetime.now() - start_time).total_seconds()
    log_success(f"Parallel phases completed in {elapsed:.1f}s")

    return drive_stats, email_stats, boston_stats, dotnews_stats


def main(parallel: bool = False):
    print_banner(f"DAILY DATA INGESTION - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if get_verbosity() >= Verbosity.VERBOSE:
        config.print_config_summary()
        config.validate_config()

    if parallel:
        drive_stats, email_stats, boston_stats, dotnews_stats = run_parallel()
    else:
        drive_stats, email_stats, boston_stats, dotnews_stats = run_sequential()

    log_run_summary(drive_stats, email_stats, boston_stats, dotnews_stats)
    print_final_summary(drive_stats, email_stats, boston_stats, dotnews_stats)

    total_errors = len(drive_stats.get("errors", [])) + len(email_stats.get("errors", [])) + (sum(len(d.get("errors", [])) for d in boston_stats.get("datasets", [])) if boston_stats else 0) + (len(dotnews_stats.get("errors", [])) if dotnews_stats else 0)

    return 1 if total_errors > 0 else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily data ingestion script")
    parser.add_argument("--parallel", "-p", action="store_true", help="Run independent phases in parallel (default: quiet logging)")
    parser.add_argument("--serial", "-s", action="store_true", help="Force serial execution (default: info logging)")
    parser.add_argument(
        "--log-level",
        "-l",
        choices=["quiet", "info", "debug"],
        default=None,
        help="Set log level explicitly (overrides mode defaults)",
    )
    args = parser.parse_args()

    parallel = args.parallel and not args.serial

    # Map log level strings to Verbosity enum
    log_level_map = {
        "quiet": Verbosity.QUIET,
        "info": Verbosity.NORMAL,
        "debug": Verbosity.VERBOSE,
    }

    # Set verbosity: explicit flag > mode default
    if args.log_level:
        set_verbosity(log_level_map[args.log_level])
    elif parallel:
        set_verbosity(Verbosity.QUIET)
    else:
        set_verbosity(Verbosity.NORMAL)

    try:
        exit_code = main(parallel=parallel)
        sys.exit(exit_code)
    except KeyboardInterrupt:
        log_warning("Interrupted by user. Exiting...")
        sys.exit(130)
    except Exception as e:
        log_error(f"FATAL ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
