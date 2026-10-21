"""Synthetic Grands Prix with exactly known lap times.

The tests deliberately do not read the bundled CSVs: every race is generated
from a closed-form lap-time model, the estimator is fitted on it, and the
coefficients are compared against the truth that generated the data. That
checks the estimator rather than the presence of a data file, so the suite is
fast and offline.
"""

from __future__ import annotations

import pandas as pd
import pytest

from f1sim.data import START_FUEL_KG, RaceContext

TOTAL_LAPS = 20
RACE_BASE = {"2025/alpha": 95.0, "2025/beta": 96.0}
DRIVER_SKILL = {"AAA": 0.15, "BBB": -0.10}
COMPOUND_OFFSET = {"SOFT": 0.05, "MEDIUM": 0.0, "HARD": -0.05}
FUEL_EFFECT = 0.030
DEG_LINEAR = {"SOFT": 0.10, "MEDIUM": 0.06, "HARD": 0.03}
#: Stint plan used for every generated race: MEDIUM, then SOFT, then HARD.
STINT_COMPOUNDS = ("MEDIUM", "SOFT", "HARD")
#: Each driver runs a different order.
#
#: If both drivers shared one order, compound and stint would be in one-to-one
#: correspondence and fuel would be an exact affine function of tyre age inside
#: every stint: the fuel column would then lie in the span of the compound
#: dummies and the per-compound age terms, and the fit could not separate the
#: fuel effect from the compound offsets. Swapping the order between drivers puts
#: the same compound at different fuel loads for the same tyre age, which is what
#: makes the fuel coefficient identified.
DRIVER_PLANS: dict[str, tuple[str, ...]] = {
    "AAA": STINT_COMPOUNDS,
    "BBB": ("SOFT", "MEDIUM", "HARD"),
}


def stint_starts() -> tuple[int, ...]:
    """The lap each of ``STINT_COMPOUNDS`` starts on, over a full distance.

    Absolute lap numbers matter: fuel load is a function of the absolute lap,
    and tyre age of the stint-relative one. If both were driven off the same
    relative counter the two columns would be perfectly collinear and the fit
    could not separate the fuel effect from the degradation rate.
    """
    starts: list[int] = []
    lap = 1
    for index in range(len(STINT_COMPOUNDS)):
        starts.append(lap)
        remaining_stints = len(STINT_COMPOUNDS) - index - 1
        length = (TOTAL_LAPS - lap + 1) // (remaining_stints + 1)
        lap += length
    return tuple(starts)


STINT_STARTS = stint_starts()
assert sum(
    (STINT_STARTS[index + 1] if index + 1 < len(STINT_STARTS) else TOTAL_LAPS + 1) - start
    for index, start in enumerate(STINT_STARTS)
) == TOTAL_LAPS


def true_lap_time(
    race_key: str, driver: str, compound: str, tyre_age: int, fuel_kg: float
) -> float:
    """The closed-form lap time the synthetic data is generated from."""
    return (
        RACE_BASE[race_key]
        + DRIVER_SKILL[driver]
        + COMPOUND_OFFSET[compound]
        + FUEL_EFFECT * fuel_kg
        + DEG_LINEAR[compound] * tyre_age
    )


def fuel_kg(lap_number: int) -> float:
    """Fuel remaining at the start of ``lap_number`` of a full race distance."""
    used = (lap_number - 1) / (TOTAL_LAPS - 1)
    return START_FUEL_KG * (1.0 - used)


def _race_laps(race_key: str, race_slug: str, year: int) -> pd.DataFrame:
    """Every lap of a full race distance, for both drivers, stint by stint."""
    rows: list[dict[str, object]] = []
    for driver in DRIVER_SKILL:
        compounds = DRIVER_PLANS[driver]
        for index, compound in enumerate(compounds):
            start = STINT_STARTS[index]
            stop = (
                STINT_STARTS[index + 1]
                if index + 1 < len(STINT_STARTS)
                else TOTAL_LAPS + 1
            )
            for lap in range(start, stop):
                load = fuel_kg(lap)
                rows.append(
                    {
                        "meeting_key": 900,
                        "session_key": 901,
                        "driver_number": 1 if driver == "AAA" else 2,
                        "lap_number": lap,
                        "lap_duration": true_lap_time(
                            race_key, driver, compound, lap - start + 1, load
                        ),
                        "is_pit_out_lap": False,
                        "stint_number": index + 1,
                        "tyre_age": lap - start + 1,
                        "name_acronym": driver,
                        "team_name": "Test Team",
                        "circuit": race_slug.title(),
                        "country": "Testland",
                        "year": year,
                        "race_slug": race_slug,
                        "compound": compound,
                        "fuel_kg": load,
                    }
                )
    return pd.DataFrame(rows)


def make_race(race_key: str, *, neutralised: frozenset[int] = frozenset()) -> RaceContext:
    """A Grand Prix whose lap times follow :func:`true_lap_time` exactly."""
    year, slug = race_key.split("/")
    laps = _race_laps(race_key, slug, int(year))
    pit_stops = pd.DataFrame(
        [
            {
                "driver_number": 1,
                "lap_number": 7,
                "stop_duration": 2.4,
                "pit_duration": 22.0,
                "lane_duration": 15.0,
            }
        ]
    )
    return RaceContext(
        race_slug=slug,
        year=int(year),
        circuit=slug.title(),
        country="Testland",
        meeting_key=1,
        session_key=1,
        laps=laps,
        pit_stops=pit_stops,
        neutralised_laps=neutralised,
        scheduled_laps=TOTAL_LAPS,
    )


@pytest.fixture
def races() -> dict[str, RaceContext]:
    """Two synthetic Grands Prix: the minimum a well-posed fit needs."""
    return {key: make_race(key) for key in RACE_BASE}


@pytest.fixture
def one_race() -> dict[str, RaceContext]:
    return {"2025/alpha": make_race("2025/alpha")}
