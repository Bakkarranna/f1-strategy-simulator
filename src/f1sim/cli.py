"""Command line interface: ``f1sim load|fit|plan|optimise|race|report``.

Every command reads the raw OpenF1 extracts in ``data/raw`` and prints numbers
that come back out of the model, so the terminal output and the README agree.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

from . import __version__
from .data import load_races
from .model import DegradationModel, fit_degradation, load_model
from .pit import neutralised_share
from .simulate import RaceSimulator, SimulationResult, Strategy
from .strategy import Candidate, StrategyOptimiser

LOGGER = logging.getLogger("f1sim")

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "degradation_model.json"


def _data_dir(args: argparse.Namespace) -> Path:
    return Path(args.data_dir).expanduser().resolve()


def _races(args: argparse.Namespace) -> dict[str, object]:
    if args.years:
        years = [int(y) for y in args.years.split(",") if y.strip()]
    else:
        years = None
    return load_races(_data_dir(args), years=years)


def _table(rows: list[dict], columns: list[tuple[str, str]], empty: str = "  (none)") -> str:
    """Render rows as a fixed-width text table (keeps the CLI dependency-free)."""
    if not rows:
        return empty
    headers = [label for _, label in columns]
    body = [[_cell(row.get(key)) for key, _ in columns] for row in rows]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in body)) for i in range(len(headers))
    ]
    out = ["  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))]
    out.append("  ".join("-" * w for w in widths))
    for row in body:
        out.append("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(out)


def _cell(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:,.3f}".rstrip("0").rstrip(".") if value % 1 else f"{value:,.0f}"
    return str(value)


DEFAULT_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")


def _compounds(text: str | None) -> tuple[str, ...]:
    """The pool of compounds a strategy search may pick from."""
    if not text:
        return DEFAULT_COMPOUNDS
    return tuple(part.strip().upper() for part in text.split(",") if part.strip())


def _stint_compounds(text: str | None, n_stints: int) -> tuple[str, ...]:
    """The start tyre plus one per stop, defaulting to SOFT -> MEDIUM -> HARD."""
    if text:
        return _compounds(text)
    return tuple(DEFAULT_COMPOUNDS[index % len(DEFAULT_COMPOUNDS)] for index in range(n_stints))


def _ranked_rows(ranked: list[Candidate]) -> list[dict]:
    """Ranked strategies with the gap to the winner, to the millisecond."""
    best = ranked[0].race_time_s if ranked else 0.0
    return [
        {
            "rank": index,
            "plan": candidate.strategy.describe(),
            "time": _fmt_clock(candidate.race_time_s),
            "gap": "-" if index == 1 else f"+{candidate.race_time_s - best:.3f} s",
        }
        for index, candidate in enumerate(ranked, start=1)
    ]



# --------------------------------------------------------------------- commands
def _race_key(races: dict[str, object], requested: str | None) -> str:
    """Pick a race: ``<year>/<slug>``, a bare slug, or the newest race."""
    if requested is None:
        return max(races, key=lambda k: (races[k].year, races[k].session_key))
    wanted = requested.strip().lower().replace(" ", "-")
    if requested in races:
        return requested
    slug_matches = [k for k in races if k.split("/")[-1] == wanted]
    if slug_matches:
        return slug_matches[0]
    circuit_matches = [
        k for k in races if wanted in races[k].circuit.lower().replace(" ", "-")
    ]
    if circuit_matches:
        return circuit_matches[0]
    raise SystemExit(f"unknown race {requested!r}; run `f1sim load` to list them")


def _model_target(args: argparse.Namespace) -> Path:
    """Where this invocation reads and writes its fitted model.

    ``--years`` changes which data a fit runs on, so a filtered fit is a
    different model. Caching it under the canonical name would leave the next
    unfiltered run loading a model fitted on one season, so it gets its own
    file and both survive in the cache.
    """
    path = Path(args.model)
    if args.years and path == MODEL_PATH:
        tag = "-" + re.sub(r"[^0-9]+", "-", args.years).strip("-")
        return path.with_name(f"{path.stem}{tag}{path.suffix}")
    return path


def _fitted(args: argparse.Namespace, races: dict[str, object]) -> DegradationModel:
    """The saved model, fitted on demand if it is missing or stale."""
    path = _model_target(args)
    if not args.refit and path.exists() and not args.years:
        return load_model(path)
    model = fit_degradation(races)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    return model


def _simulator(
    args: argparse.Namespace, races: dict[str, object]
) -> tuple[str, object, RaceSimulator]:
    key = _race_key(races, args.race)
    race = races[key]
    model = _fitted(args, races)
    return key, race, RaceSimulator(race, model)


def _default_stops(race: object) -> tuple[int, ...]:
    """The number of stops this race actually ran, spread over the distance."""
    laps = race.scheduled_laps or len(race.dry_laps())
    typical = 2
    if not race.pit_stops.empty and "lap_number" in race.pit_stops:
        per_car = race.pit_stops.groupby("driver_number")["lap_number"].count()
        typical = int(round(float(per_car.median())))
    typical = max(1, min(typical, 3))
    # Stop on every multiple of the ideal even split, so a two-stop race stops
    # at 1/3 and 2/3 of the distance rather than at the same lap twice.
    window = laps / float(typical + 1)
    return tuple(
        max(1, min(int(round(window * (index + 1))), laps - 1))
        for index in range(typical)
    )


def _cmd_load(args: argparse.Namespace) -> int:
    races = _races(args)
    rows = []
    for key, race in sorted(races.items()):
        rows.append(
            {
                "race": key,
                "circuit": race.circuit,
                "laps": len(race.dry_laps()),
                "total": race.scheduled_laps,
                "neutralised": len(race.neutralised_laps),
            }
        )
    args.out.write(
        f"{len(races)} races, {sum(r['laps'] for r in rows)} clean green-flag laps, "
        f"{neutralised_share(races):.1%} of the race distance run neutralised\n"
    )
    args.out.write(_table(rows, [("race", "race"), ("circuit", "circuit"), ("laps", "green-flag laps"), ("total", "race laps"), ("neutralised", "SC/VSC laps")]) + "\n")
    return 0


def _cmd_fit(args: argparse.Namespace) -> int:
    races = _races(args)
    model = fit_degradation(races)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = model.save(args.model)
    args.out.write(f"fitted on {model.training_laps} green-flag laps\n")

    args.out.write(f"  fuel effect      {model.fuel_effect:+.4f} s/kg\n")
    args.out.write(f"  residual sigma   {model.residual_sigma:.3f} s\n")
    args.out.write(f"  pit-lane loss    {model.pit_loss_s:.2f} s (measured)\n")
    args.out.write(f"  SC rate / lap    {model.sc_rate_per_lap:.5f}\n")
    args.out.write("  degradation      " + ", ".join(f"{c} {model.deg_lin[c]:+.3f} s/lap" for c in model.compound_offset) + "\n")
    args.out.write(f"saved {path}\n")
    return 0


# ---------------------------------------------------------------- plan command
def _cmd_plan(args: argparse.Namespace) -> int:
    races = _races(args)
    key, race, sim = _simulator(args, races)
    stops = (
        tuple(int(x) for x in args.stops.split(",") if x.strip())
        if args.stops
        else _default_stops(race)
    )
    compounds = _stint_compounds(args.compounds, len(stops) + 1)
    if len(compounds) != len(stops) + 1:
        raise SystemExit(
            f"{len(stops)} stops need {len(stops) + 1} compounds, "
            f"got {len(compounds)}: {', '.join(compounds)}"
        )
    strategy = Strategy(stops=stops, compounds=compounds)
    result = sim.simulate_single(strategy)
    out = args.out
    out.write(f"{key}  {race.label}  {sim.total_laps} laps\n")
    out.write(f"  plan             {strategy.describe()}\n")
    out.write(f"  pit-lane loss    {sim.pit_loss_s:.1f} s x {strategy.n_stops} stops\n")
    out.write(f"  race time        {_fmt_clock(result.total_time_s)}\n")
    out.write(f"  average lap      {sum(result.lap_times) / len(result.lap_times):.3f} s\n")
    out.write(f"  fastest lap      {min(result.lap_times):.3f} s\n")
    out.write(
        "  stints           "
        + " | ".join(
            f"L{start}-{start + length - 1} {compound} ({length} laps)"
            for start, length, compound in strategy.stint_plan(sim.total_laps)
        )
        + "\n"
    )
    return 0


def _cmd_optimise(args: argparse.Namespace) -> int:
    races = _races(args)
    key, race, sim = _simulator(args, races)
    ranked = StrategyOptimiser(sim).best(
        max_stops=args.max_stops,
        min_stint=args.min_stint,
        compounds=_compounds(args.compounds),
        top=args.top,
    )
    if not ranked:
        raise SystemExit("no legal strategy found at this circuit")
    args.out.write(
        f"{key}  {race.label}  {sim.total_laps} laps  pit loss {sim.pit_loss_s:.1f} s\n"
    )
    args.out.write(
        _table(
            [
                {
                    "rank": index,
                    "plan": candidate.strategy.describe(),
                    "time": _fmt_clock(candidate.race_time_s),
                }
                for index, candidate in enumerate(ranked, start=1)
            ],
            [("rank", "#"), ("plan", "strategy"), ("time", "race time"), ("gap", "gap")],
        )
        + "\n"
    )
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACTS_DIR / f"strategies_{race.race_slug}.json"
    out_path.write_text(
        json.dumps(
            {
                "race": key,
                "circuit": race.circuit,
                "total_laps": sim.total_laps,
                "pit_loss_s": round(sim.pit_loss_s, 2),
                "strategies": [candidate.as_dict() for candidate in ranked],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    args.out.write(f"saved {out_path}\n")
    return 0


# ----------------------------------------------------------------+ race command
def _cmd_race(args: argparse.Namespace) -> int:
    races = _races(args)
    key, race, sim = _simulator(args, races)
    ranked = StrategyOptimiser(sim).best(
        max_stops=args.max_stops,
        min_stint=args.min_stint,
        compounds=_compounds(args.compounds),
        top=args.field,
    )
    if not ranked:
        raise SystemExit("no legal strategy found at this circuit")
    plans = [candidate.strategy for candidate in ranked]
    results = sim.simulate_race(
        plans[0], plans=plans, n_sims=args.sims, field_size=len(plans), seed=args.seed
    )
    tally = _tally(results, plans)
    args.out.write(
        f"{key}  {race.label}  Monte Carlo over {args.sims} races, "
        f"{len(plans)} strategies, seed {args.seed}\n"
    )
    rows = []
    for index, plan in enumerate(plans):
        wins, podium, mean = tally[index]
        rows.append(
            {
                "plan": plan.describe(),
                "wins": f"{wins / args.sims:.1%}",
                "podium": f"{podium / args.sims:.1%}",
                "mean": f"{mean:.2f}",
            }
        )
    args.out.write(
        _table(
            rows,
            [("plan", "strategy"), ("wins", "win %"), ("podium", "podium %"), ("mean", "mean pos")],
        )
        + "\n"
    )
    return 0


def _tally(
    results: list[SimulationResult], plans: list[Strategy]
) -> list[tuple[int, int, float]]:
    """Per-plan ``(wins, podiums, mean position)`` over a Monte Carlo field."""
    stats = [[0, 0, 0.0] for _ in plans]
    counts = [0] * len(plans)
    for result in results:
        index = next(
            (i for i, plan in enumerate(plans) if plan == result.strategy), None
        )
        if index is None:
            continue
        stats[index][0] += 1 if result.position == 1 else 0
        stats[index][1] += 1 if (result.position or 99) <= 3 else 0
        stats[index][2] += float(result.position or 0)
        counts[index] += 1
    return [
        (wins, podium, total / count if count else 0.0)
        for (wins, podium, total), count in zip(stats, counts, strict=True)
    ]


# -------------------------------------------------------------- report command
def _cmd_report(args: argparse.Namespace) -> int:
    races = _races(args)
    model = _fitted(args, races)
    key, race, sim = _simulator(args, races)
    ranked = StrategyOptimiser(sim).best(
        max_stops=args.max_stops,
        min_stint=args.min_stint,
        compounds=_compounds(args.compounds),
        top=args.top,
    )
    out_path = REPO_ROOT / "reports" / "RESULTS.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        _report_markdown(key, race, sim, model, ranked, _validation(races, model, args)),
        encoding="utf-8",
    )
    args.out.write(f"wrote {out_path}\n")
    return 0


def _validation(races: dict[str, object], model: DegradationModel, args: argparse.Namespace) -> dict[str, float]:
    """Leave-races-out scores for the model being reported.

    A report that only quotes in-sample numbers claims the model is right
    because it fitted the data it was fitted on. Holding out whole races and
    refitting the rest costs one extra fit per fold, which is affordable here
    and is the number a reader should trust.
    """
    if args.cv_folds <= 1 or len(races) < 3:
        return {"folds": 0, "races": 0, "races_skipped": len(races)}
    return model.cross_validate(races, folds=args.cv_folds)


def _validation_lines(validation: dict[str, float]) -> list[str]:
    """The held-out numbers, or an explicit note that they were not measured."""
    folds = int(validation.get("folds", 0))
    races = int(validation.get("races", 0))
    if folds < 2 or races == 0:
        return [
            "## Validation",
            "",
            "Not measured: leave-races-out validation needs at least three",
            "circuits and two folds.",
            "",
        ]
    skipped = int(validation.get("races_skipped", 0))
    scored_laps = int(validation.get("laps", 0))
    signed = validation["mean_error_s"]
    mae = validation["mae_s"]
    rmse = validation["rmse_s"]
    skipped_note = f" ({skipped} skipped: no clean laps)" if skipped else ""
    return [
        "## Validation (held-out circuits)",
        "",
        f"Each of {folds} folds refits the model on the other circuits and then",
        "measures the held-out ones as if they were a brand-new Grand Prix,",
        "calibrating their baseline from their own clean laps. This scores the",
        "fuel and degradation terms rather than memorising each circuit.",
        "",
        f"- circuits scored: {races}{skipped_note}",
        f"- clean laps scored: {scored_laps}",
        f"- mean signed error: {_signed_seconds(signed)}",
        f"- mean absolute error: {mae:.3f} s",
        f"- rmse: {rmse:.3f} s",
        "",
    ]


def _report_markdown(
    key: str,
    race: object,
    sim: RaceSimulator,
    model: DegradationModel,
    ranked: list[Candidate],
    validation: dict[str, float] | None = None,
) -> str:
    lines = [
        f"# Strategy report - {race.label}",
        "",
        "Generated by `f1sim report` from the raw OpenF1 extracts in `data/raw`.",
        "",
        "## The fitted lap-time model",
        "",

        f"- race: {key} ({race.circuit}), {sim.total_laps} laps",
        f"- training laps: {model.training_laps}",
        f"- fuel effect: {model.fuel_effect:+.4f} s/kg",
        f"- residual sigma: {model.residual_sigma:.3f} s",
        f"- measured pit-lane loss: {model.pit_loss_s:.2f} s",
        f"- safety-car rate: {model.sc_rate_per_lap:.5f} per lap",
        "",
        "Compound degradation, relative to the fastest compound:",
        "",
    ]
    fastest = min(model.compound_offset.values())
    for compound, offset in sorted(model.compound_offset.items()):
        lines.append(
            f"- **{compound}** {offset - fastest:+.3f} s intrinsic, "
            f"{model.deg_lin[compound]:+.4f} s per lap of tyre age"
        )
    lines += [
        "",
        *_validation_lines(validation or {}),
        "## Ranked strategies",
        "",
        _table(
            _ranked_rows(ranked),
            [
                ("rank", "#"),
                ("plan", "strategy"),
                ("time", "race time"),
                ("gap", "gap to best"),
            ],
        ),
        "",
    ]
    return "\n".join(lines)


def _fmt_clock(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 3600}:{total % 3600 // 60:02d}:{total % 60:02d}"


def _signed_seconds(value: float) -> str:
    """A signed second count.

    ``f"{value:+.3f}"`` prints ``-0.000`` for any error small enough to round
    away, and a minus sign in front of a zero reads as a formatting bug rather
    than as "no bias".
    """
    text = f"{value:+.3f}"
    if float(text) == 0.0 and text.startswith("-"):
        text = f"+{text[1:]}"
    return f"{text} s"


# ---------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="f1sim",
        description="Formula 1 pit-stop strategy simulator and Monte Carlo race model.",
    )
    parser.add_argument("--version", action="version", version=f"f1sim {__version__}")
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    shared.add_argument("--years", default=None, help="comma-separated years, e.g. 2024,2025")
    shared.add_argument("--race", default=None, help="<year>/<slug>, slug, or circuit name")
    shared.add_argument("--compounds", default=None, help="start tyre plus one per stop")
    shared.add_argument("--model", default=str(MODEL_PATH))
    shared.add_argument("--refit", action="store_true")
    shared.add_argument("--top", type=int, default=10)
    shared.add_argument("--max-stops", type=int, default=2)
    shared.add_argument("--min-stint", type=int, default=8)
    shared.add_argument("--field", type=int, default=8)
    shared.add_argument("--sims", type=int, default=200)
    shared.add_argument(
        "--cv-folds",
        type=int,
        default=2,
        help="leave-races-out folds used to score a model (2 = one extra fit)",
    )
    shared.add_argument("--seed", type=int, default=0)
    shared.add_argument("--stops", default=None, help="comma-separated stop laps")
    sub = parser.add_subparsers(dest="command", metavar="command")
    sub.add_parser("load", parents=[shared]).set_defaults(func=_cmd_load)
    sub.add_parser("fit", parents=[shared]).set_defaults(func=_cmd_fit)
    sub.add_parser("plan", parents=[shared]).set_defaults(func=_cmd_plan)
    sub.add_parser("optimise", parents=[shared]).set_defaults(func=_cmd_optimise)
    sub.add_parser("race", parents=[shared]).set_defaults(func=_cmd_race)
    sub.add_parser("report", parents=[shared]).set_defaults(func=_cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point behind the ``f1sim`` console script."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    args.out = sys.stdout
    return args.func(args)
