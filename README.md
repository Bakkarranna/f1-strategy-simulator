# f1-strategy-simulator — pit-stop strategy and Monte Carlo race model

A lap-time model fitted on real Formula 1 lap data, plus a race simulator that
turns a pit-stop plan into a predicted race time. Everything that drives a
strategy decision is measured from the data rather than assumed: tyre
degradation per compound, the fuel effect, the pit-lane time loss per circuit,
and how often each circuit is neutralised.

## What it models

One clean racing lap is modelled as

```
lap_time = circuit_base(race)
         + driver_skill(driver)
         + compound_offset(compound)
         + fuel_effect * fuel_kg
         + deg_lin[compound] * tyre_age
         + deg_sq[compound] * tyre_age^2
```

Fitting is a sparse least-squares solve (`scipy.sparse.linalg.lsqr`) over
one-hot encoded races and drivers, so every coefficient is identifiable and
readable: a compound offset is a real speed difference, a degradation
coefficient is real seconds per lap of tyre age.

The simulator then runs a strategy lap by lap — which compound, how old the
tyre is, how much fuel is left — and charges one pit-lane loss per stop.

Two levels of simulation are provided:

- `simulate_single` — one car on an empty track. Deterministic, fast, and the
  right tool for comparing strategies against each other.
- `simulate_race` — a Monte Carlo of a whole field, where every car gets its
  own pace noise, a safety car can appear on any lap, and cars can be held up
  by slower cars. This answers questions like "does the undercut still work
  after a safety car?" or "what is the probability of a podium?".

Only clean green-flag laps are fitted or simulated, so the model never takes
credit for laps that were run under a safety car.

## Validation

A model quoted only by how well it fits the laps it was fitted on proves
nothing, so the report scores it the way it will actually be used. Holding out
every circuit in turn, refitting the model on the rest, and only then scoring
the unseen circuit — re-measuring that circuit's own baseline first, so what
is being tested is the fuel and degradation terms rather than a memorised
intercept:

```
in-sample residual sigma   0.834 s
held-out RMSE             0.852 s   (47 scored circuits, 41,408 clean laps)
```

Those are the numbers a strategy decision can stand on: 2% more error on
circuits the model has never met. `f1sim report` regenerates both from
`data/raw` and writes them to [reports/RESULTS.md](reports/RESULTS.md).

## Installation

```bash
python -m pip install -e ".[dev]"
```

Requires Python 3.11+, numpy, pandas, scipy and requests.

## Data

The bundled extracts in `data/raw/` cover the 2024 and 2025 seasons
(48 Grands Prix, 41,408 clean green-flag laps) and were downloaded from the
[OpenF1](https://openf1.org) API by `scripts/fetch_laps.py`. To refresh them:

```bash
python scripts/fetch_laps.py --years 2024 2025
```

Responses are cached under `data/cache/openf1/`, so re-runs are cheap.

## Usage

```bash
f1sim load                       # list every race, its lap count and SC laps
f1sim fit                        # fit the lap-time model and save it
f1sim plan                       # race time for one pit-stop plan
f1sim optimise                   # search for the fastest legal plan
f1sim race --sims 1000           # Monte Carlo of a whole field
f1sim report                     # write reports/RESULTS.md
```

A race is selected with `--race <year>/<slug>`, a bare slug, or a circuit
name; without it the most recent race is used. Other useful flags:
`--compounds SOFT,MEDIUM`, `--stops 20,40`, `--top 10`, `--field 20`,
`--seed 7`, `--refit`.

Example:

```bash
f1sim optimise --race 2025/monza --max-stops 2 --top 10
```

## Repository layout

```
src/f1sim/data.py       race loading, cleaning and safety-car detection
src/f1sim/model.py      the lap-time model: fitting, prediction, validation
src/f1sim/simulate.py   deterministic and Monte Carlo race simulation
src/f1sim/strategy.py   candidate strategy enumeration and ranking
src/f1sim/pit.py        measured pit-lane loss and neutralisation rates
src/f1sim/cli.py        the f1sim command line interface
scripts/fetch_laps.py   OpenF1 downloader for the raw extracts
tests/                  unit tests on synthetic races with known answers
```

## Tests

```bash
pytest -q
ruff check .
```

The tests build synthetic Grands Prix whose lap times follow a known formula,
fit the model on them, and assert the coefficients come back. That checks the
estimator itself rather than the presence of data files, so the suite runs
offline and in seconds.

## License

MIT — see [LICENSE](LICENSE).
