"""Loading and cleaning of the raw OpenF1 extracts.

The raw CSVs (see ``scripts/fetch_laps.py``) come from the OpenF1 API. This
module turns them into the tidy structures the models use:

* ``RaceContext.laps``       - one row per racing lap, with fuel and tyre state
* ``RaceContext.pit_stops``  - one row per pit stop with measured time loss
* ``RaceContext.neutralised``- laps run under Safety Car / VSC
* ``RaceContext.meta``       - circuit, race name and scheduled lap count

Cleaning rules (documented in the README, covered by tests):

* lap 1 is dropped: it is a standing start from a standstill
* in-laps and out-laps of pit stops are dropped (pit-lane speed limit)
* laps that are more than ``OUTLIER_FACTOR`` x the race median are dropped
  (traffic, driver errors, damaged cars, slow-down laps)
* laps run under a neutralised (yellow / safety car) flag state are dropped
* wet-weather running is dropped: on inters the concept of a dry
  degradation curve does not apply
:mod:`f1pit`-style note: ``tyre_age`` from OpenF1 is the number of laps
completed on that tyre *within* the stint (it restarts at 1 on a fresh set),
which is exactly the age the degradation curve needs. ``tyre_age_at_start``
is a separate OpenF1 field that stays 0 for most sessions, so it is ignored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# Columns kept from the raw lap file.
LAP_COLUMNS = [
    "meeting_key",
    "session_key",
    "driver_number",
    "lap_number",
    "lap_duration",
    "duration_sector_1",
    "duration_sector_2",
    "duration_sector_3",
    "is_pit_out_lap",
    "st_speed",
    "compound",
    "stint_number",
    "tyre_age_at_start",
    "tyre_age",
    "name_acronym",
    "team_name",
    "circuit",
    "country",
    "year",
    "race_slug",
]

DRY_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
OUTLIER_FACTOR = 1.075
START_FUEL_KG = 110.0
FUEL_EFFECT_S_PER_KG = 0.035  # overwritten by the fitted value when available
#: A safety-car phase is capped at this many laps so that a message logging gap
#: cannot neutralise the rest of the race.
MAX_SC_LAPS = 8
#: A sector yellow is assumed to hold for at most this many laps.
MAX_YELLOW_LAPS = 4


@dataclass
class RaceContext:
    """Everything the models need to know about one Grand Prix."""

    race_slug: str
    year: int
    circuit: str
    country: str
    meeting_key: int
    session_key: int
    laps: pd.DataFrame
    pit_stops: pd.DataFrame
    neutralised_laps: frozenset[int]
    scheduled_laps: int
    fuel_lap_lost: pd.Series = field(default_factory=pd.Series)

    @property
    def key(self) -> str:
        """The key the fitted model stores this race under: ``<year>/<slug>``.

        ``race_slug`` alone repeats across seasons (two Italian Grands Prix ran
        at Imola), so the model builds its race intercepts on year-and-slug.
        Always look a race up in a model with this value, never the bare slug.
        """
        return f"{self.year}/{self.race_slug}"

    @property
    def label(self) -> str:
        return f"{self.year} {self.circuit}"

    def total_laps(self) -> int:
        return int(self.scheduled_laps)

    def dry_laps(self) -> pd.DataFrame:
        return self.laps[self.laps["compound"].isin(DRY_COMPOUNDS)]


def _neutralised_laps(control: pd.DataFrame) -> frozenset[int]:
    """Laps whose time was affected by a safety car, VSC or a yellow flag.

    OpenF1's race-control stream is a log of messages, each stamped with the
    lap it was issued on. Two families of message make a lap unusable for a
    tyre-degradation fit:

    * **Safety car / VSC phases** - "SAFETY CAR DEPLOYED", "VSC DEPLOYED" and
      "SAFETY CAR IN THIS LAP". A phase is closed by the next message that
      says the track is clear again (a ``GREEN`` / ``CLEAR`` flag event, or a
      "... ENDING" message). Every lap inside the phase is neutralised.
    * **Yellow flags** - "YELLOW" / "DOUBLE YELLOW" raised on a sector. The
      flag holds from the lap it was raised until the matching
      "CLEAR IN TRACK SECTOR n" for that same sector, so those laps are
      neutralised too: a driver lifts through a yellow sector.

    Blue flags (a lapped car being waved past) and black-and-white flags do
    not slow the whole field, so those laps are kept.
    """
    if control.empty:
        return frozenset()
    control = control.sort_values("date").reset_index(drop=True)
    control["lap_number"] = pd.to_numeric(control["lap_number"], errors="coerce")
    text = control["message"].fillna("").str.upper()
    flags = control["flag"].fillna("").str.upper()
    neutralised: set[int] = set()

    # --- safety car and virtual safety car phases -------------------------
    deploy = text.str.contains(
        r"SAFETY CAR DEPLOYED|VSC DEPLOYED|VIRTUAL SAFETY CAR DEPLOYED|"
        r"SAFETY CAR IN THIS LAP|SAFETY CAR THROUGH THE PIT LANE",
        regex=True,
    )
    end = text.str.contains(
        r"SAFETY CAR ENDING|VIRTUAL SAFETY CAR ENDING|VSC ENDING|END OF SAFETY CAR PERIOD",
        regex=True,
    )
    green = control["category"].fillna("").str.upper().eq("FLAG") & flags.isin(["GREEN", "CLEAR"])
    deploy_flags = control["category"].fillna("").str.upper().eq("SAFETYCAR") & deploy

    pending_start: int | None = None
    for idx, row in control.iterrows():
        lap = row["lap_number"]
        if pd.isna(lap):
            continue
        lap = int(lap)
        if deploy_flags.iloc[idx] or deploy.iloc[idx]:
            if pending_start is None:
                pending_start = lap
            continue
        if pending_start is not None and (end.iloc[idx] or green.iloc[idx]):
            if lap >= pending_start:
                neutralised.update(range(pending_start, min(lap, pending_start + MAX_SC_LAPS) + 1))
                pending_start = None
    if pending_start is not None:
        # The phase was never explicitly closed: assume it ran on until the
        # end of the race, capped so a logging gap cannot swallow the race.
        neutralised.update(range(pending_start, min(control["lap_number"].max(), pending_start + MAX_SC_LAPS) + 1))

    # --- sector yellow flags ----------------------------------------------
    yellow = flags.isin(["YELLOW", "DOUBLE YELLOW"])
    clear = flags.eq("CLEAR") & control["scope"].fillna("").str.upper().eq("SECTOR")
    sectors = control["sector"]
    open_yellow: dict[str, int] = {}
    for idx, row in control.iterrows():
        lap = row["lap_number"]
        if pd.isna(lap):
            continue
        lap = int(lap)
        sector = str(sectors.iloc[idx])
        if yellow.iloc[idx] and sector not in ("nan", ""):
            open_yellow[sector] = lap
        elif clear.iloc[idx] and sector in open_yellow:
            start = open_yellow.pop(sector)
            neutralised.update(range(start, min(lap, start + MAX_YELLOW_LAPS) + 1))
    for start in open_yellow.values():
        neutralised.update(range(start, min(start + MAX_YELLOW_LAPS, start + 2) + 1))

    return frozenset(lap for lap in neutralised if lap > 0)


def _clean_laps(laps: pd.DataFrame, neutralised: frozenset[int]) -> pd.DataFrame:
    keep = [c for c in LAP_COLUMNS if c in laps.columns]
    laps = laps[keep].copy()
    laps = laps[laps["compound"].isin(DRY_COMPOUNDS)]
    laps = laps[laps["lap_duration"].notna()]
    laps = laps[laps["lap_number"] > 1]  # standing start
    laps = laps[~laps["is_pit_out_lap"].fillna(False)]
    if neutralised:
        laps = laps[~laps["lap_number"].isin(neutralised)]
    if laps.empty:
        return laps

    # In-lap: the lap on which a driver pitted. OpenF1 flags the *out* lap, so
    # detect the in-lap as the last lap of a stint.
    last_of_stint = laps.groupby(["driver_number", "stint_number"])["lap_number"].transform("max")
    laps = laps[laps["lap_number"] != last_of_stint]

    median = laps["lap_duration"].median()
    laps = laps[laps["lap_duration"] <= median * OUTLIER_FACTOR]
    laps = laps.sort_values(["driver_number", "lap_number"]).reset_index(drop=True)
    laps = laps.assign(
        fuel_kg=START_FUEL_KG * (1.0 - (laps["lap_number"] - 1) / (laps.groupby("race_slug")["lap_number"].transform("max") - 1))
    )
    return laps


_FRAME_CACHE: dict[tuple[str, str], pd.DataFrame] = {}


def _read_cached(data_dir: Path, filename: str) -> pd.DataFrame:
    """Read a raw extract once per process.

    ``load_races`` walks every race in a file; without this cache each race
    would re-parse the whole season's lap file (48 files x 2 = 96 full reads).
    """
    key = (str(data_dir), filename)
    frame = _FRAME_CACHE.get(key)
    if frame is None:
        frame = pd.read_csv(data_dir / "raw" / filename, low_memory=False)
        _FRAME_CACHE[key] = frame
    return frame


def load_race(
    data_dir: str | Path,
    year: int,
    race_slug: str,
    race_laps: dict[str, int] | None = None,
    control: pd.DataFrame | None = None,
) -> RaceContext:
    """Load one Grand Prix from the raw CSV extracts."""
    data_dir = Path(data_dir)
    laps = _read_cached(data_dir, f"laps_{year}.csv")
    laps = laps[laps["race_slug"] == race_slug]
    pit = _read_cached(data_dir, f"pit_{year}.csv")
    pit = pit[pit["session_key"].isin(laps["session_key"].unique())]
    sessions = _read_cached(data_dir, f"sessions_{year}.csv")

    session_key = int(laps["session_key"].iloc[0])
    if control is None:
        control_path = data_dir / "raw" / f"control_{year}.csv"
        control = (
            pd.read_csv(control_path, low_memory=False)
            if control_path.exists()
            else pd.DataFrame()
        )
    control = control[control["session_key"] == session_key] if not control.empty else control
    neutralised = _neutralised_laps(control)

    if laps.empty:
        raise ValueError(f"no laps found for {year}/{race_slug}")
    meta = sessions[sessions["session_key"] == session_key]
    circuit = str(meta["circuit_short_name"].iloc[0]) if not meta.empty else race_slug
    country = str(meta["country_name"].iloc[0]) if not meta.empty else ""

    observed = int(laps["lap_number"].max())
    scheduled = int((race_laps or {}).get(race_slug, observed))

    return RaceContext(
        race_slug=race_slug,
        year=year,
        circuit=circuit,
        country=country,
        meeting_key=int(laps["meeting_key"].iloc[0]),
        session_key=session_key,
        laps=_clean_laps(laps, neutralised),
        pit_stops=pit,
        neutralised_laps=neutralised,
        scheduled_laps=scheduled,
    )


def load_races(data_dir: str | Path, years: list[int] | None = None) -> dict[str, RaceContext]:
    """Load every race in the raw extracts, keyed by ``<year>/<race_slug>``."""
    data_dir = Path(data_dir)
    years = years or sorted(
        int(p.stem.split("_")[1])
        for p in (data_dir / "raw").glob("laps_*.csv")
    )
    races: dict[str, RaceContext] = {}
    for year in years:
        path = data_dir / "raw" / f"laps_{year}.csv"
        if not path.exists():
            continue
        control = (
            _read_cached(data_dir, f"control_{year}.csv")
            if (data_dir / "raw" / f"control_{year}.csv").exists()
            else pd.DataFrame()
        )
        laps = _read_cached(data_dir, f"laps_{year}.csv")
        for slug in sorted(laps["race_slug"].unique()):
            races[f"{year}/{slug}"] = load_race(data_dir, year, slug, control=control)
    return races
