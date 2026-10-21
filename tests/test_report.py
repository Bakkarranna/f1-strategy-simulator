"""The report the project hands to a reader, and what it must not claim.

The model tests already check that the estimator is right. These check the
document that *publishes* the estimator: a report that quoted only in-sample
accuracy would claim the model works because it fitted the data it was fitted
on, and a reader cannot tell the difference by looking at the prose.
"""

from __future__ import annotations

from f1sim.cli import _report_markdown, _validation_lines
from f1sim.model import DegradationModel
from f1sim.simulate import RaceSimulator
from f1sim.strategy import Candidate, Strategy

#: A fold result whose signed mean is negative but too small to survive three
#: decimals. That is the case where ``f"{x:+.3f}"`` prints ``-0.000``.
UNBIASED_FOLD = {
    "folds": 2,
    "races": 2,
    "races_skipped": 0,
    "laps": 80,
    "mean_error_s": -1e-9,
    "mae_s": 0.596,
    "rmse_s": 0.852,
}


def test_the_report_publishes_held_out_numbers(races):
    """The validation numbers have to be reachable in the written report."""
    model = DegradationModel.fit(races)
    race = races["2025/alpha"]
    sim = RaceSimulator(race, model)
    ranked = [
        Candidate(
            strategy=Strategy(stops=(10,), compounds=("MEDIUM", "HARD")),
            race_time_s=2400.0,
        )
    ]

    text = _report_markdown("2025/alpha", race, sim, model, ranked, UNBIASED_FOLD)

    assert "## Validation (held-out circuits)" in text
    assert "mean absolute error: 0.596 s" in text
    assert "rmse: 0.852 s" in text
    # between the fitted coefficients and the ranked strategies, so a reader
    # meets them before any strategy recommendation
    assert text.index("Validation") < text.index("## Ranked strategies")


def test_a_full_report_still_carries_the_validation_section(races):
    """Guards the default ``validation=None`` path from silently dropping it."""
    model = DegradationModel.fit(races)
    race = races["2025/alpha"]

    text = _report_markdown("2025/alpha", race, RaceSimulator(race, model), model, [], {})

    assert "Validation" in text


def test_unmeasured_validation_says_so_instead_of_vanishing(races):
    """No folds must read as an admission, not as an absence of the section."""
    lines = _validation_lines({"folds": 0, "races": 0, "races_skipped": 2})
    text = "\n".join(lines)

    assert "Not measured" in text
    assert "## Validation" in text


def test_a_mean_error_that_rounds_to_zero_loses_its_minus_sign():
    """""-0.000`` reads like a formatting bug; ``+0.000`` reads like no bias."""
    lines = _validation_lines(UNBIASED_FOLD)

    assert "- mean signed error: +0.000 s" in lines
    assert not any("-0.000" in line for line in lines)
