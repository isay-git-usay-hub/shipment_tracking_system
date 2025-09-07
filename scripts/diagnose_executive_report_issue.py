#!/usr/bin/env python
"""
Diagnose why the Executive Summary report returns "No data found for specified criteria".

What it does:
- Reads DB (SQLite) to compute dataset min/max timestamp and total records
- Counts records within a given UI date range
- If the API is reachable, calls /reports/executive-summary for that range to confirm

Usage:
  python scripts/diagnose_executive_report_issue.py --start YYYY-MM-DD --end YYYY-MM-DD
If --start/--end are omitted, it defaults to the last 30 days.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, date
import json
import sys
from pathlib import Path

# Ensure project root is on import path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import func, and_  # type: ignore

try:
    from config.settings import settings
    from core.database import get_db_session
    from core.models import Shipment
except Exception as e:
    print(f"Import error: {e}")
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(description="Diagnose Executive Summary 'No data found' issue")
    p.add_argument("--start", type=str, default=None, help="UI start date YYYY-MM-DD")
    p.add_argument("--end", type=str, default=None, help="UI end date YYYY-MM-DD (inclusive)")
    p.add_argument("--api", action="store_true", help="Also call API /reports/executive-summary if reachable")
    return p.parse_args()


def to_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def default_range() -> tuple[date, date]:
    end = datetime.utcnow().date()
    start = end - timedelta(days=30)
    return start, end


def db_stats(ui_start: date, ui_end: date) -> dict:
    with get_db_session() as db:
        # Overall dataset stats
        min_ts, max_ts, total = db.query(
            func.min(Shipment.timestamp),
            func.max(Shipment.timestamp),
            func.count(Shipment.id),
        ).one()

        # Count within UI range (inclusive end)
        ui_end_dt = datetime.combine(ui_end, datetime.max.time())
        ui_start_dt = datetime.combine(ui_start, datetime.min.time())
        in_range = db.query(func.count(Shipment.id)).filter(
            and_(Shipment.timestamp >= ui_start_dt, Shipment.timestamp <= ui_end_dt)
        ).scalar() or 0

        return {
            "dataset": {
                "total_records": int(total or 0),
                "min_timestamp": min_ts.isoformat() if min_ts else None,
                "max_timestamp": max_ts.isoformat() if max_ts else None,
            },
            "ui_range": {
                "start": ui_start.isoformat(),
                "end": ui_end.isoformat(),
                "records_in_range": int(in_range),
            },
        }


def try_call_api(ui_start: date, ui_end: date) -> dict:
    try:
        import requests  # type: ignore
    except Exception:
        return {"api_call": "skipped (requests not available)"}

    base = f"http://{settings.API_HOST}:{settings.API_PORT}"
    # Health
    try:
        r = requests.get(f"{base}/health", timeout=3)
        health_ok = r.ok
    except Exception as e:
        return {"api_reachable": False, "error": f"health check failed: {e}"}

    payload = {
        "start_date": ui_start.isoformat(),
        "end_date": ui_end.isoformat(),
        "period_days": (ui_end - ui_start).days + 1,
    }
    try:
        r = requests.post(f"{base}/reports/executive-summary", json=payload, timeout=10)
        return {
            "api_reachable": health_ok,
            "status_code": r.status_code,
            "response": r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text,
        }
    except Exception as e:
        return {"api_reachable": health_ok, "error": str(e)}


def conclusion(stats: dict) -> str:
    ds = stats["dataset"]
    ui = stats["ui_range"]
    if ds["min_timestamp"] and ds["max_timestamp"]:
        ds_start = datetime.fromisoformat(ds["min_timestamp"]).date()
        ds_end = datetime.fromisoformat(ds["max_timestamp"]).date()
        if ui["records_in_range"] == 0:
            if ui["end"] < ds["min_timestamp"] or ui["start"] > ds["max_timestamp"]:
                return (
                    "No shipments in the selected date range. The selected UI range is outside the dataset range "
                    f"({ds_start} to {ds_end}). Adjust the dashboard date filter to fall within this range."
                )
            else:
                return (
                    "No shipments match the current filters within the date range. Verify asset filter and other conditions."
                )
        else:
            return "Records exist in the date range; if API still returns no data, investigate request payload or backend filtering."
    return "Dataset appears empty; load data into the database."


def main():
    args = parse_args()
    if args.start and args.end:
        ui_start, ui_end = to_date(args.start), to_date(args.end)
    else:
        ui_start, ui_end = default_range()

    stats = db_stats(ui_start, ui_end)
    result = {
        "settings": {
            "api_host": settings.API_HOST,
            "api_port": settings.API_PORT,
            "database_url": settings.DATABASE_URL,
        },
        "stats": stats,
        "conclusion": conclusion(stats),
    }

    if args.api:
        api_result = try_call_api(ui_start, ui_end)
        result["api_test"] = api_result

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

