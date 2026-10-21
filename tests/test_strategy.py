"""Pit-stop plans: the rules a plan has to obey, and how it splits a race."""

from __future__ import annotations

import pytest

from f1sim.simulate import Strategy


def test_a_plan_needs_one_compound_per_stint():
    # two stops means three stints: the starting tyre plus one per stop
    with pytest.raises(ValueError, match="needs"):
        Strategy(stops=(10,), compounds=("SOFT",))


def test_stops_must_increase():
    with pytest.raises(ValueError, match="strictly increasing"):
        Strategy(stops=(12, 10), compounds=("SOFT", "MEDIUM", "HARD"))


def test_stops_must_be_positive_laps():
    with pytest.raises(ValueError, match="positive"):
        Strategy(stops=(0,), compounds=("SOFT", "MEDIUM"))


def test_no_stops_is_one_stint_for_the_whole_distance():
    plan = Strategy(stops=(), compounds=("HARD",))
    assert plan.stint_plan(53) == [(1, 53, "HARD")]


def test_stints_cover_the_distance_exactly_once():
    plan = Strategy(stops=(17, 35), compounds=("SOFT", "MEDIUM", "HARD"))
    stints = plan.stint_plan(53)
    assert [(start, length) for start, length, _ in stints] == [
        (1, 17),
        (18, 18),
        (36, 18),
    ]
    assert sum(length for _, length, _ in stints) == 53
    # compound per stint: the start tyre, then one per stop
    assert [compound for _, _, compound in stints] == ["SOFT", "MEDIUM", "HARD"]


def test_a_stop_on_the_final_lap_adds_no_stint():
    # the stop lap is the last lap, so there is no second stint to run
    plan = Strategy(stops=(10,), compounds=("SOFT", "HARD"))
    assert plan.stint_plan(10) == [(1, 10, "SOFT")]


def test_describe_reads_like_a_race_engineer():
    assert Strategy(stops=(26,), compounds=("HARD", "MEDIUM")).describe() == (
        "start on HARD, lap 26 -> MEDIUM"
    )
    assert Strategy(stops=(), compounds=("SOFT",)).describe() == "no stops (SOFT)"
