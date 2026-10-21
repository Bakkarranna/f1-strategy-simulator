"""Download the race data the strategy simulator is fitted on.

Usage::

    python scripts/fetch_laps.py                    # 2025 season, all races
    python scripts/fetch_laps.py --years 2024 2025  # several seasons
    python scripts/fetch_laps.py --dry              # list races, download nothing
    python scripts/fetch_laps.py --refresh          # ignore the local cache

Writes one CSV per season under ``data/raw/``::

    laps_<year>.csv        one row per lap (join of laps, stints, drivers)
    drivers_<year>.csv     driver number -> name / team mapping
    pit_<year>.csv         pit stop events with pit lane transit time
    control_<year>.csv     race control messages (safety car, flags, red flag)
    sessions_<year>.csv    session metadata: circuit, country, start time

Responses are cached under ``data/cache/openf1/`` so re-runs are cheap. Only
``race`` sessions are fetched, because the simulator models a race distance.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.openf1.org/v1"
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
CACHE_DIR = ROOT / "data" / "cache" / "openf1"

TIMEOUT = 90
MIN_DELAY = 0.25  # seconds between calls, to stay inside the API rate limit
PAGE = 1000


def _get(path: str, *, refresh: bool = False, **params) -> list[dict]:
    """GET one OpenF1 endpoint, caching the decoded rows on disk."""
    key = hashlib.sha1(
        json.dumps([path, sorted(params.items())], sort_keys=True).encode()
    ).hexdigest()[:16]
    cache_path = CACHE_DIR / f"{path.replace('/', '_')}_{key}.json.gz"

    if not refresh and cache_path.exists():
        try:
            return json.loads(gzip.decompress(cache_path.read_bytes()).decode())
        except (OSError, ValueError):
            pass

    for attempt in range(5):
        try:
            time.sleep(MIN_DELAY)
            response = requests.get(f"{BASE}/{path}", params=params, timeout=TIMEOUT)
            if response.status_code == 404:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_bytes(gzip.compress(b"[]"))
                return []
            if response.status_code == 429 or response.status_code >= 500:
                time.sleep(2 + 2 * attempt)
                continue
            response.raise_for_status()
            payload = response.json()
            rows = payload if isinstance(payload, list) else []
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(
                gzip.compress(json.dumps(rows).encode("utf-8"))
            )
            return rows
        except (requests.RequestException, ValueError):
            if attempt == 4:
                raise
            time.sleep(2 + 2 * attempt)
    return []


def _get_all(path: str, *, refresh: bool = False, **params) -> list[dict]:
    """Deprecated shim kept for callers that used the paginated helper.

    OpenF1 returns 404 for any request that carries ``limit``/``offset``, so
    paging is not supported; this simply issues one request.
    """
    return _get(path, refresh=refresh, **params)


PAGE_UNUSED = """OpenF1 answers 404 when ``limit``/``offset`` are supplied, so every
endpoint is fetched in one request and the API is polled slowly instead."""


def race_sessions(year: int) -> list[dict]:
    """Session metadata for every Grand Prix of a season."""
    rows = _get("sessions", year=year, session_name="Race")
    return sorted(rows, key=lambda row: row.get("date_start") or "")


def season(year: int, *, refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Download every race of a season and join the pieces together."""
    sessions = race_sessions(year)
    if not sessions:
        print(f"  {year}: no race sessions found", file=sys.stderr)
        return {}

    lap_rows: list[pd.DataFrame] = []
    driver_rows: list[pd.DataFrame] = []
    pit_rows: list[pd.DataFrame] = []
    control_rows: list[pd.DataFrame] = []

    for index, session in enumerate(sessions, start=1):
        key = int(session["session_key"])
        laps = pd.DataFrame(_get("laps", refresh=refresh, session_key=key))
        if laps.empty:
            print(f"  {year} {index}/{len(sessions)}: {key} has no laps", file=sys.stderr)
            continue
        stints = pd.DataFrame(_get("stints", refresh=refresh, session_key=key))
        drivers = pd.DataFrame(_get("drivers", refresh=refresh, session_key=key))
        pit = pd.DataFrame(_get("pit", refresh=refresh, session_key=key))
        control = pd.DataFrame(_get("race_control", refresh=refresh, session_key=key))

        # The laps endpoint no longer carries stint or compound information, so
        # the stint table is joined on the lap range each stint covers and the
        # tyre age is derived from where the lap falls inside that range.
        if not stints.empty:
            wanted = [
                "driver_number",
                "compound",
                "lap_start",
                "lap_end",
                "stint_number",
                "tyre_age_at_start",
            ]
            stints = stints[[c for c in wanted if c in stints.columns]]
            merged = laps.merge(stints, on="driver_number", how="left")
            inside = (
                merged["lap_number"].ge(merged["lap_start"])
                & merged["lap_number"].le(merged["lap_end"])
            )
            merged = merged[inside]
            merged["tyre_age"] = (
                merged["lap_number"]
                - merged["lap_start"]
                + 1
                + merged["tyre_age_at_start"]
            ).astype("Int64")
            laps = merged.drop(columns=["lap_start", "lap_end"])
        if not drivers.empty:
            keep = ["driver_number", "name_acronym", "full_name", "team_name"]
            laps = laps.merge(
                drivers[[c for c in keep if c in drivers.columns]],
                on="driver_number",
                how="left",
            )
            drivers = drivers.assign(session_key=key)

        laps["session_key"] = key
        laps["meeting_key"] = session.get("meeting_key")
        laps["circuit"] = session.get("circuit_short_name")
        laps["country"] = session.get("country_name")
        laps["year"] = session.get("year")
        laps["race_slug"] = f"{session.get('circuit_short_name', 'race')}".lower()
        lap_rows.append(laps)

        if not pit.empty:
            pit["session_key"] = key
            pit_rows.append(pit)
        if not control.empty:
            control["session_key"] = key
            control_rows.append(control)
        driver_rows.append(drivers if not drivers.empty else pd.DataFrame())

        print(
            f"  {year} {index}/{len(sessions)}: {session.get('circuit_short_name')} "
            f"({len(laps)} laps)",
            file=sys.stderr,
        )

    out = {
        "laps": pd.concat(lap_rows, ignore_index=True) if lap_rows else pd.DataFrame(),
        "drivers": pd.concat(driver_rows, ignore_index=True)
        if driver_rows
        else pd.DataFrame(),
        "pit": pd.concat(pit_rows, ignore_index=True) if pit_rows else pd.DataFrame(),
        "control": pd.concat(control_rows, ignore_index=True)
        if control_rows
        else pd.DataFrame(),
        "sessions": pd.DataFrame(sessions),
    }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=[2025])
    parser.add_argument("--refresh", action="store_true", help="ignore the cache")
    parser.add_argument("--dry", action="store_true", help="only list the races")
    args = parser.parse_args(argv)

    for year in args.years:
        print(f"{year}:", file=sys.stderr)
        sessions = race_sessions(year)
        for session in sessions:
            print(
                f"  round {session.get('round', '?')}"
                f" {session.get('circuit_short_name')}: key={session['session_key']}",
                file=sys.stderr,
            )
        if args.dry:
            continue

        tables = season(year, refresh=args.refresh)
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        for name, frame in tables.items():
            if frame.empty:
                continue
            path = RAW_DIR / (f"{name}_{year}.csv" if name != "sessions" else f"sessions_{year}.csv")
            frame.to_csv(path, index=False)
            print(f"  wrote {path} ({len(frame)} rows)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
