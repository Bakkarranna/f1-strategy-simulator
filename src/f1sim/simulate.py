"""Turning a pit-stop plan into a predicted race time.

The simulator is deliberately transparent. For a given plan it walks the race
lap by lap, asking the fitted lap-time model for each lap:

* which compound the car is on,
* how old those tyres are,
* how much fuel is left,
* plus a once-per-race pit-lane time loss for every stop.

Two levels of simulation are provided:

``simulate_single``
    A single car running the plan on an empty road. Fast, deterministic, and
    the right tool for comparing strategies against each other.

``simulate_race``
    A Monte Carlo of a whole field, where every car gets its own pace noise,
    a safety car can appear at any lap, and cars can be stuck behind slower
    cars. This is what you need for questions like "is the undercut still
    worth it after a safety car?" or "what is the probability this finishes
    on the podium?".

Only the *clean* laps from the fitted model are used, so the simulator never
takes credit for laps that were run under a safety car.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data import START_FUEL_KG, RaceContext
from .model import DegradationModel

#: Losing this much relative pace costs you roughly this much traffic time.
TRAFFIC_PENALTY_S_PER_LAP = 0.35
TRAFFIC_PASS_LAPS = 1.5
#: Fraction of a lap that is spent in the pit lane for a typical circuit.
PIT_LANE_FRACTION = 1.0
#: Default safety car probability per lap, used when no circuit data exists.
DEFAULT_SC_RATE_PER_LAP = 0.02
DEFAULT_SC_LAPS = 3
#: Under a safety car a car loses this much of its degradation-driven loss.
SC_DEGRADATION_FACTOR = 0.55
#: A pit stop taken while the Safety Car is out costs this fraction of the
#: green-flag pit-lane loss, because the field is already running slowly.
SC_PIT_FACTOR = 0.4


@dataclass(frozen=True)
class Strategy:
    """A pit-stop plan: where to stop and on what tyre."""

    stops: tuple[int, ...]
    compounds: tuple[str, ...]

    def __post_init__(self) -> None:
        # One compound per *stint*, and a plan with N stops runs N + 1
        # stints: the opening tyre plus one for each stop.
        if len(self.compounds) != len(self.stops) + 1:
            raise ValueError(
                f"a strategy with {len(self.stops)} stop(s) needs "
                f"{len(self.stops) + 1} compounds (the starting tyre plus one "
                f"per stop), got {len(self.compounds)}"
            )
        if any(b <= a for a, b in zip(self.stops, self.stops[1:], strict=False)):
            raise ValueError(f"stops must be strictly increasing, got {self.stops}")
        if not all(isinstance(s, int) and s > 0 for s in self.stops):
            raise ValueError(f"stops must be positive lap numbers, got {self.stops}")

    @property
    def n_stops(self) -> int:
        return len(self.stops)

    def stint_plan(self, total_laps: int) -> list[tuple[int, int, str]]:
        """Split the race into ``(start_lap, laps_on_tyre, compound)`` stints."""
        compounds = self.compounds
        if not self.stops:
            return [(1, total_laps, compounds[0])]
        bounds = [1] + [lap + 1 for lap in self.stops] + [total_laps + 1]
        stints: list[tuple[int, int, str]] = []
        for index, start in enumerate(bounds[:-1]):
            length = bounds[index + 1] - start
            if length > 0:
                stints.append((start, length, compounds[index]))
        return stints

    def describe(self) -> str:
        parts = [f"start on {self.compounds[0]}"]
        for lap, compound in zip(self.stops, self.compounds[1:], strict=False):
            parts.append(f"lap {lap} -> {compound}")
        if not self.stops:
            parts = [f"no stops ({self.compounds[0]})"]
        return ", ".join(parts)

    def as_dict(self) -> dict[str, object]:
        return {"stops": list(self.stops), "compounds": list(self.compounds)}


@dataclass
class SimulationResult:
    """Outcome of one simulated race."""

    strategy: Strategy
    total_time_s: float
    lap_times: list[float] = field(default_factory=list)
    position: int | None = None
    safety_car_laps: tuple[int, ...] = ()
    gap_to_leader_s: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "strategy": self.strategy.as_dict(),
            "total_time_s": round(self.total_time_s, 3),
            "position": self.position,
            "gap_to_leader_s": None if self.gap_to_leader_s is None else round(self.gap_to_leader_s, 3),
            "safety_car_laps": list(self.safety_car_laps),
        }


class RaceSimulator:
    """Simulates a race at a given circuit using a fitted lap-time model.

    Parameters
    ----------
    race:
        The :class:`~f1sim.data.RaceContext` being simulated.
    model:
        A fitted :class:`~f1sim.model.DegradationModel`.
    pit_loss_s:
        Measured pit-lane time loss; defaults to the measured mean.
    sc_rate_per_lap / sc_laps:
        Safety-car frequency and length used by the Monte Carlo.
    """

    def __init__(
        self,
        race: RaceContext,
        model: DegradationModel,
        pit_loss_s: float | None = None,
        sc_rate_per_lap: float | None = None,
        sc_laps: int = DEFAULT_SC_LAPS,
    ) -> None:
        self.race = race
        self.model = model
        self.pit_loss_s = pit_loss_s if pit_loss_s is not None else model.pit_loss_s
        self.sc_rate_per_lap = (
            sc_rate_per_lap if sc_rate_per_lap is not None else model.sc_rate_per_lap
        )
        self.sc_laps = sc_laps
        self.total_laps = race.scheduled_laps or int(race.laps["lap_number"].max())

    # ------------------------------------------------------------------ laps
    def fuel_kg(self, lap_number: int) -> float:
        """Fuel remaining at the start of ``lap_number`` of a full race."""
        used = (lap_number - 1) / max(self.total_laps - 1, 1)
        return START_FUEL_KG * (1.0 - used)

    def lap_time(
        self,
        lap_number: int,
        compound: str,
        tyre_age: int,
        driver: str | None = None,
        safety_car: bool = False,
    ) -> float:
        """Model lap time, with an optional safety-car correction."""
        # The model fits one intercept per *season-and-circuit*, so the lookup
        # key is the race's key, not its slug.
        reference = self.model.predict(
            self.race.key,
            compound,
            tyre_age,
            self.fuel_kg(lap_number),
            driver=driver,
        )
        if safety_car:
            # Under a safety car the field runs to a delta: roughly 1.4x a
            # green-flag lap, and the degradation penalty is mostly erased
            # because the tyres are working far less hard.
            green_floor = self.model.circuit_floor(self.race.key)
            neutral = max(green_floor * 1.35, self.model.predict(
                self.race.key,
                compound,
                tyre_age * SC_DEGRADATION_FACTOR,
                self.fuel_kg(lap_number),
                driver=driver,
            ) * 0.92)
            return neutral
        return reference

    def stint_lap_times(self, strategy: Strategy, driver: str | None = None) -> tuple[list[float], list[int]]:
        """Lap-by-lap times for one car running ``strategy``."""
        times: list[float] = []
        ages: list[int] = []
        for start, length, compound in strategy.stint_plan(self.total_laps):
            for lap in range(start, start + length):
                if lap > self.total_laps:
                    break
                age = lap - start + 1
                ages.append(age)
                times.append(self.lap_time(lap, compound, age, driver=driver))
        return times, ages

    # ------------------------------------------------------- single-car race
    def simulate_single(
        self,
        strategy: Strategy,
        driver: str | None = None,
        lap_number_start: int = 1,
    ) -> SimulationResult:
        """Deterministic race time for one car on an empty track."""
        times, _ = self.stint_lap_times(strategy, driver=driver)
        loss = self.pit_loss_s * strategy.n_stops
        return SimulationResult(
            strategy=strategy,
            total_time_s=float(sum(times) + loss),
            lap_times=times,
        )

    # --------------------------------------------------------- full-field race
    def _safety_car_laps(self, rng: np.random.Generator, field_size: int) -> set[int]:
        """Which laps of this simulated race run under a Safety Car.

        The rate is the share of laps this circuit actually loses to
        neutralisation, so Monaco is not simulated with Spa's safety-car
        frequency.
        """
        per_race = 1.0 - (1.0 - self.sc_rate_per_lap) ** self.total_laps
        if per_race <= 0.0 or rng.random() >= per_race:
            return set()
        latest = max(self.total_laps - self.sc_laps, 1)
        start = int(rng.integers(1, latest + 1))
        return set(range(start, min(start + self.sc_laps, self.total_laps) + 1))

    def simulate_race(
        self,
        strategy: Strategy,
        *,
        plans: list[Strategy] | None = None,
        n_sims: int = 1000,
        field_size: int = 20,
        seed: int = 0,
    ) -> list[SimulationResult]:
        """Monte Carlo of a whole field, one result per simulation.

        Every car runs a plan (``plans[i]`` for car ``i``, or ``strategy`` for
        all of them when no list is given), and three things differ between
        simulations and between cars:

        *pace*
            Each car draws an offset from the model's residual sigma, so the
            field spreads out the way the residuals in the fitted data do.

        *safety cars*
            A Safety Car can appear on any lap, drawn from this circuit's
            measured neutralisation rate and lasting ``sc_laps``. A stop taken
            while the Safety Car is out costs a fraction of the green-flag
            pit-lane loss, which is why the field always pits when it can.

        *traffic*
            A car only loses time in the wake of slower cars: the gap it holds
            to the car in front decides how much of the race it spends in
            dirty air, and the loss per lap is capped at
            :data:`TRAFFIC_PENALTY_S_PER_LAP`. A car in clear air loses
            nothing.

        The result is a *distribution*, not a point estimate: run it twice and
        sort by race time to answer "how often does this plan win?" instead of
        "what is this plan's lap time?".
        """
        rng = np.random.default_rng(seed)
        sigma = max(self.model.residual_sigma, 1e-6)
        field = [strategy for _ in range(field_size)]
        if plans:
            field = list(plans)
            field_size = len(field)
        outcomes: list[SimulationResult] = []
        for _ in range(n_sims):
            pace_offset = {
                index: float(rng.normal(0.0, sigma)) for index in range(field_size)
            }
            sc_laps = self._safety_car_laps(rng, field_size)
            fastest = min(pace_offset.values())
            times: dict[int, float] = {}
            for index, plan in enumerate(field):
                lap_times = [
                    self.lap_time(
                        lap,
                        compound,
                        age,
                        safety_car=lap in sc_laps,
                    )
                    + pace_offset[index]
                    for lap, (compound, age) in _plan_laps(plan, self.total_laps)
                ]
                pit_time = 0.0
                for stop_lap in plan.stops:
                    cheap = stop_lap in sc_laps
                    pit_time += self.pit_loss_s * (SC_PIT_FACTOR if cheap else 1.0)
                traffic = self._traffic_loss(
                    pace_offset[index] - fastest, field_size, rng
                ) * self.total_laps
                times[index] = float(sum(lap_times) + pit_time + traffic)
            order = sorted(times, key=lambda i: times[i])
            leader = times[order[0]]
            for position, index in enumerate(order, start=1):
                outcomes.append(
                    SimulationResult(
                        strategy=field[index],
                        total_time_s=times[index],
                        position=position,
                        safety_car_laps=tuple(sorted(sc_laps)),
                        gap_to_leader_s=times[index] - leader,
                    )
                )
        return outcomes

    def _traffic_loss(
        self, pace_gap_s: float, field_size: int, rng: np.random.Generator
    ) -> float:
        """Expected dirty-air loss per lap for a car holding ``pace_gap_s``.

        A car no slower than the quickest car in the field runs in clear air.
        Beyond that the share of the race spent following someone grows with
        the gap, and the loss per lap is drawn up to
        :data:`TRAFFIC_PENALTY_S_PER_LAP`.
        """
        if pace_gap_s <= 0.0:
            return 0.0
        share = min(pace_gap_s / (TRAFFIC_PASS_LAPS * max(field_size, 2)), 0.6)
        if float(rng.random()) >= share:
            return 0.0
        return TRAFFIC_PENALTY_S_PER_LAP * float(rng.random())


def _plan_laps(plan: Strategy, total_laps: int) -> list[tuple[int, tuple[str, int]]]:
    """``(lap_number, (compound, tyre_age))`` for one plan, pit stops included."""
    out: list[tuple[int, tuple[str, int]]] = []
    stop_laps = set(plan.stops)
    stint = 0
    age = 0
    for lap in range(1, total_laps + 1):
        if lap in stop_laps:
            stint += 1
            age = 0
        age += 1
        out.append((lap, (plan.compounds[min(stint, len(plan.compounds) - 1)], age)))
    return out
