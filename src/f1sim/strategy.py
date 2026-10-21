"""Searching for the fastest pit-stop plan.

The optimiser enumerates candidate strategies - every sensible combination of
stop count, stop laps and compounds - and scores each one with the race
simulator. Rather than a black-box optimiser, this is a deliberate,
inspectable search:

* the candidate ladder is built from the actual pit-stop windows seen in the
  data (so the search is bounded by what is physically possible given a
  minimum and maximum stint length),
* every candidate is scored with the deterministic simulator,
* the best few are then re-scored with the Monte Carlo to see how they behave
  when a safety car appears.
"""

from __future__ import annotations

from dataclasses import dataclass

from .simulate import RaceSimulator, Strategy


@dataclass
class Candidate:
    """One strategy with its deterministic cost."""

    strategy: Strategy
    race_time_s: float

    def as_dict(self) -> dict[str, object]:
        return {
            "strategy": self.strategy.describe(),
            "stops": self.strategy.as_dict()["stops"],
            "compounds": self.strategy.as_dict()["compounds"],
            "race_time_s": round(self.race_time_s, 3),
        }


class StrategyOptimiser:
    """Enumerate and rank strategies for one circuit."""

    def __init__(self, simulator: RaceSimulator) -> None:
        self.sim = simulator
        self.total_laps = simulator.total_laps

    def candidates(
        self,
        max_stops: int = 3,
        min_stint: int = 8,
        compounds: tuple[str, ...] = ("SOFT", "MEDIUM", "HARD"),
    ) -> list[Candidate]:
        """All legal strategies up to ``max_stops`` stops."""
        results: list[Candidate] = []
        for n_stops in range(1, max_stops + 1):
            for plan in self._stop_combinations(n_stops, min_stint):
                for tyres in self._compound_combinations(n_stops, compounds):
                    strategy = Strategy(stops=tuple(plan), compounds=tuple(tyres))
                    if not self._is_legal(strategy, min_stint):
                        continue
                    results.append(
                        Candidate(strategy=strategy, race_time_s=self.sim.simulate_single(strategy).total_time_s)
                    )
        results.sort(key=lambda c: c.race_time_s)
        return results

    def best(
        self,
        max_stops: int = 3,
        min_stint: int = 8,
        compounds: tuple[str, ...] = ("SOFT", "MEDIUM", "HARD"),
        top: int = 10,
    ) -> list[Candidate]:
        return self.candidates(max_stops, min_stint, compounds)[:top]

    def _stop_combinations(self, n_stops: int, min_stint: int) -> list[list[int]]:
        """Every increasing list of stop laps with at least one stint each."""
        if n_stops == 0:
            return [[]]
        choices: list[list[int]] = []

        def walk(start: int, chosen: list[int]) -> None:
            if len(chosen) == n_stops:
                choices.append(list(chosen))
                return
            remaining = n_stops - len(chosen)
            first_lap = start
            last_lap = self.total_laps - remaining * min_stint
            for lap in range(first_lap, last_lap + 1):
                chosen.append(lap)
                walk(lap + min_stint, chosen)
                chosen.pop()

        walk(1, [])
        return choices

    def _compound_combinations(self, n_stops: int, compounds: tuple[str, ...]) -> list[list[str]]:
        """Start tyre plus one tyre per stop, with no immediate repeats."""
        combos: list[list[str]] = []
        for start in compounds:
            for rest in _product(compounds, n_stops):
                if any(rest[i] == rest[i - 1] for i in range(1, len(rest))):
                    continue
                if rest and rest[0] == start:
                    continue
                combos.append([start, *rest])
        return combos

    def _is_legal(self, strategy: Strategy, min_stint: int) -> bool:
        stints = strategy.stint_plan(self.total_laps)
        if not stints:
            return False
        if any(length < min_stint for _, length, _ in stints):
            return False
        if strategy.stops[-1] >= self.total_laps:
            return False
        return True


def _product(values: tuple[str, ...], count: int) -> list[tuple[str, ...]]:
    if count == 0:
        return [()]
    out: list[tuple[str, ...]] = []
    for value in values:
        for tail in _product(values, count - 1):
            out.append((value, *tail))
    return out
