#!/usr/bin/env python3
"\"\"\"Convert Zeek JSON logs to CSV with a fixed header.\"\"\""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence


JST = timezone(timedelta(hours=9))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Zeek JSON logs in a directory into a single CSV file.",
    )
    parser.add_argument(
        "log_dir",
        type=Path,
        help="Directory that contains Zeek JSON log files (e.g., conn.log).",
    )
    parser.add_argument(
        "output_csv",
        type=Path,
        help="Path to the CSV file that will be written.",
    )
    parser.add_argument(
        "--pattern",
        default="*.log",
        help="Glob pattern used to find log files (default: %(default)s).",
    )
    return parser.parse_args()


def find_log_files(log_dir: Path, pattern: str) -> List[Path]:
    files = sorted(p for p in log_dir.glob(pattern) if p.is_file())
    if not files:
        raise SystemExit(f"No log files matching '{pattern}' under {log_dir}")
    return files


def iter_records(files: Sequence[Path]) -> Iterator[dict]:
    for log_file in files:
        with log_file.open("r", encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    yield json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise SystemExit(
                        f"Invalid JSON in {log_file}:{line_number}: {exc}"
                    ) from exc


def collect_header(records: Iterable[dict]) -> List[str]:
    seen = set()
    header: List[str] = []
    for record in records:
        for key in record.keys():
            normalized_key = "daytime" if key == "ts" else key
            if normalized_key not in seen:
                seen.add(normalized_key)
                header.append(normalized_key)
    if not header:
        raise SystemExit("No JSON objects were found in the provided log files.")
    return header


def convert_ts_to_daytime(value) -> str:
    if value is None or value == "":
        return ""
    try:
        ts = float(value)
    except (TypeError, ValueError):
        return ""
    utc_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return utc_dt.astimezone(JST).strftime("%Y-%m-%d %H:%M:%S")


def normalize_value(value):
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=True)
    return value


def write_csv(files: Sequence[Path], header: Sequence[str], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for record in iter_records(files):
            row = {}
            for key in header:
                if key == "daytime":
                    row[key] = convert_ts_to_daytime(record.get("ts", record.get("daytime", "")))
                else:
                    row[key] = normalize_value(record.get(key, ""))
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    if not args.log_dir.is_dir():
        raise SystemExit(f"Log directory not found: {args.log_dir}")

    files = find_log_files(args.log_dir, args.pattern)

    # Collect header from a first pass over the logs so that the CSV schema is fixed.
    header = collect_header(iter_records(files))

    # Re-read the files to emit the CSV (iter_records provides fresh iterators each call).
    write_csv(files, header, args.output_csv)


if __name__ == "__main__":
    main()
