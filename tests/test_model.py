"""The estimator itself: coefficients, identifiability, and validation.

``conftest`` generates synthetic Grands Prix from a closed-form lap time, so a
correct fit must return the coefficients that generated the data. That checks
the estimator, not the presence of a data file.
"""

from __future__ import annotations

import numpy as np
import pytest

from f1sim.data import RaceContext
from f1sim.model import DegradationModel, FitReport, _design_frame

from .conftest import (
    COMPOUND_OFFSET,
    DEG_LINEAR,
    DRIVER_SKILL,
    FUEL_EFFECT,
    make_race,
)


def test_fit_recovers_the_coefficients_that_generated_the_data(races):
    model = DegradationModel.fit(races)

    assert model.fuel_effect == pytest.approx(FUEL_EFFECT, abs=0.005)
    for compound, offset in COMPOUND_OFFSET.items():
        assert model.compound_offset[compound] == pytest.approx(offset, abs=0.005)
    for compound, slope in DEG_LINEAR.items():
        assert model.deg_lin[compound] == pytest.approx(slope, abs=0.005)


def test_driver_skills_are_contrasts_not_an_absorbed_constant(races):
    """Skills are contrasts around the field, so each stays within ~0.05 s.

    The design drops one reference level per categorical block to stay full
    rank. If the dropped level's constant were not handed back, the fit would
    dump it into whichever block kept its coefficients, and a driver "skill"
    would come out as tens of seconds: still fitting, still wrong.
    """
    model = DegradationModel.fit(races)
    for driver, skill in DRIVER_SKILL.items():
        assert model.driver_skill[driver] == pytest.approx(skill, abs=0.05)

    # contrasts must sum to zero, which is what pins the reference level
    assert sum(model.driver_skill.values()) == pytest.approx(0.0, abs=1e-6)


def test_re_centring_the_blocks_moves_no_fitted_lap_time(races):
    model = DegradationModel.fit(races)
    for key, race in races.items():
        base = model.calibrate_race_base(race.laps)
        predicted = model.predict_frame(race.laps, key, race_base=base)
        actual = race.laps["lap_duration"].to_numpy(dtype=float)
        assert predicted.shape == actual.shape
        assert np.abs(predicted - actual).max() < 0.01


def test_a_race_with_no_clean_laps_is_dropped_not_fitted(races):
    """2024/interlagos was so neutralised that no lap survived cleaning."""
    races["2025/empty"] = _empty_race()

    frame = _design_frame(races)
    assert "2025/empty" not in set(frame["race"])

    model = DegradationModel.fit(races)
    assert "2025/empty" not in model.race_base


def test_cross_validation_ignores_a_race_with_no_clean_laps(races):
    races["2025/empty"] = _empty_race()
    model = DegradationModel.fit(races)

    report = model.cross_validate(races, folds=2)

    assert report["folds"] == 2
    assert report["races"] == len(races) - 1
    assert report["races_skipped"] == 1
    # the synthetic data is exact, so held-out errors must be tiny
    assert report["mae_s"] < 0.05
    assert report["rmse_s"] < 0.05
    assert np.isfinite(report["mean_error_s"])


def test_cross_validation_scores_unseen_circuits_honestly(races):
    """A held-out circuit is scored only after its base is re-measured."""
    model = DegradationModel.fit(races)
    report = model.cross_validate(races, folds=2)
    # with a calibrated intercept, only the true fuel/degradation error remains
    assert report["races"] == len(races)
    assert abs(report["mean_error_s"]) < 0.05


def test_a_fold_that_trains_on_nothing_is_dropped_not_crashed(races):
    """``folds=1`` leaves no training race, so there is nothing to measure."""
    model = DegradationModel.fit(races)
    assert np.isnan(model.cross_validate(races, folds=1)["mae_s"])


def test_fit_report_pairs_in_sample_and_cross_validated_metrics(races):
    report = FitReport.build(races, folds=2)
    summary = report.as_dict()
    assert summary["summary"]["training_laps"] == 80
    assert summary["cross_validation"]["races"] == 2


def test_an_unseen_circuit_gets_the_field_mean_not_a_crash(races):
    model = DegradationModel.fit(races)
    base = model.race_base_value("2099/mars-gp")
    assert base == pytest.approx(float(np.mean(list(model.race_base.values()))))


def test_a_model_with_no_fitted_intercepts_refuses_to_guess_one():
    bare = DegradationModel(
        fuel_effect=0.03,
        compound_offset={},
        deg_lin={},
        deg_sq={},
        race_base={},
        driver_skill={},
        residual_sigma=0.0,
        training_laps=0,
    )
    with pytest.raises(ValueError, match="no fitted race intercepts"):
        bare.race_base_value("2099/mars-gp")


def _empty_race() -> RaceContext:
    """A Grand Prix whose every lap was run under a safety car."""
    template = make_race("2025/alpha")
    return RaceContext(
        race_slug="interlagos",
        year=2024,
        circuit="Interlagos",
        country="Brazil",
        meeting_key=template.meeting_key,
        session_key=template.session_key,
        laps=template.laps.iloc[0:0],
        pit_stops=template.pit_stops.iloc[0:0],
        neutralised_laps=frozenset(range(1, 21)),
        scheduled_laps=20,
    )
