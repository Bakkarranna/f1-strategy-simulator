"""F1 race strategy simulator: tyre degradation modelling and Monte Carlo races.

Public API
---------
load_race          -> load a single Grand Prix as a ``RaceContext``
fit_degradation    -> fit the lap-time model across a season of races
RaceModel          -> predict lap time from tyre age, compound, fuel and driver
RaceSimulator      -> simulate a race, deterministically or as a Monte Carlo
strategy.optimise   -> search for the fastest pit-stop plan for a race
"""

from .data import RaceContext, load_race, load_races
from .model import (
    DegradationModel,
    FitReport,
    fit_degradation,
    load_model,
    tidy_laps,
)
from .simulate import RaceSimulator, SimulationResult, Strategy
from .strategy import StrategyOptimiser

__all__ = [
    "DegradationModel",
    "FitReport",
    "RaceContext",
    "RaceSimulator",
    "SimulationResult",
    "Strategy",
    "StrategyOptimiser",
    "fit_degradation",
    "load_model",
    "load_race",
    "load_races",
    "tidy_laps",
]

__version__ = "1.0.0"
