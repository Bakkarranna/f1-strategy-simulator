"""Pit-lane time loss and safety-car frequency, measured from the real data.

Two numbers drive most of the strategy conversation, and both are measured
here rather than assumed:

``pit_loss``
    How much time a pit stop actually costs relative to staying out. It is
    measured as (in-lap + out-lap) minus two typical green-flag laps at the
    same point in the race, per circuit. The OpenF1 ``pit`` table also carries
    ``stop_duration`` (stationary time) and ``pit_duration`` (time in the pit
    lane); those are reported separately because they answer different
    questions.

``neutralisation``
    The share of laps actually run under a safety car or VSC, per circuit.
"""

from __future__ import annotations

import pandas as pd


def pit_loss_by_circuit(races: dict[str, object]) -> pd.DataFrame:
    """Measured pit-lane loss per circuit, in seconds."""
    rows: list[dict[str, float | str]] = []
    for key, race in races.items():
        laps = race.laps
        if laps.empty or race.pit_stops.empty:
            continue
        per_lap = laps.set_index(["driver_number", "lap_number"])["lap_duration"]
        for driver, group in race.pit_stops.groupby("driver_number"):
            for lap in sorted(int(x) for x in group["lap_number"].tolist()):
                in_lap = per_lap.get((driver, lap))
                out_lap = per_lap.get((driver, lap + 1))
                if in_lap is None or out_lap is None:
                    continue
                window = laps[
                    (laps["driver_number"] == driver)
                    & (laps["lap_number"].between(max(2, lap - 4), lap - 1))
                ]["lap_duration"]
                if len(window) < 2:
                    continue
                reference = float(window.median()) * 2
                rows.append(
                    {
                        "race_key": key,
                        "circuit": race.circuit,
                        "loss_s": float(in_lap + out_lap - reference),
                        "stop_duration": float(group["stop_duration"].iloc[0]),
                        "pit_duration": float(group["pit_duration"].iloc[0]),
                        "lane_duration": float(group["lane_duration"].iloc[0]),
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["circuit", "loss_s", "stops"])
    frame = pd.DataFrame(rows)
    frame = frame[frame["loss_s"].between(0, 60)]
    return (
        frame.groupby("circuit")
        .agg(loss_s=("loss_s", "mean"), stops=("loss_s", "size"), stop_time_s=("stop_duration", "mean"))
        .reset_index()
        .sort_values("loss_s")
    )


def neutralisation_summary(races: dict[str, object]) -> pd.DataFrame:
    """How often each circuit is neutralised, and for how many laps."""
    rows = []
    for key, race in races.items():
        total = race.scheduled_laps or int(race.laps["lap_number"].max())
        rows.append(
            {
                "race_key": key,
                "circuit": race.circuit,
                "year": race.year,
                "neutralised_laps": len(race.neutralised_laps),
                "total_laps": total,
                "neutralised_share": len(race.neutralised_laps) / max(total, 1),
            }
        )
    return pd.DataFrame(rows)


def neutralised_share(races: dict[str, object]) -> float:
    """Share of all scheduled laps that were run neutralised."""
    summary = neutralisation_summary(races)
    if summary.empty:
        return 0.0
    return float(summary["neutralised_laps"].sum() / summary["total_laps"].sum())


def default_pit_loss(races: dict[str, object]) -> float:
    """A single pit-loss number used when a circuit has too few samples."""
    summary = pit_loss_by_circuit(races)
    if summary.empty:
        return 21.0
    return float(summary["loss_s"].mean())
