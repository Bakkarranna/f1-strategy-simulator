"""The lap-time model that sits under the race simulator.

One clean racing lap is modelled as

    lap_time = circuit_base(race)
             + driver_skill(driver)
             + compound_offset(compound)
             + fuel_effect * fuel_kg
             + deg_lin[compound] * tyre_age
             + deg_sq[compound] * tyre_age^2

Every term has a physical or sporting interpretation, which is what makes the
model useful for strategy questions rather than only for prediction:

* ``circuit_base``   - how long the lap takes at that circuit in those
  conditions (includes track evolution, temperature, wind)
* ``driver_skill``   - the driver's underlying pace relative to the field
* ``compound_offset``- the intrinsic speed difference between the compounds
* ``fuel_effect``    - seconds per kg of fuel, which is the reason a car gets
  quicker through a stint even when the tyres are not degrading
* ``deg_lin``/``deg_sq`` - linear and quadratic tyre degradation per lap

Fitting uses a sparse least-squares solve on one-hot encoded races and
drivers (``scipy.sparse.linalg.lsqr``). Holding out whole races and refitting
only the race intercept gives an honest estimate of how the model behaves on a
circuit it has never seen - see ``DegradationModel.cross_validate``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr

from . import pit
from .data import DRY_COMPOUNDS, START_FUEL_KG

LOGGER = logging.getLogger(__name__)

#: Default safety-car rate per lap, used when no circuit data is available.
DEFAULT_SC_RATE = 0.02

#: Target column of the model.
TARGET = "lap_duration"


@dataclass
class DegradationModel:
    """A fitted lap-time model, serialisable to and from JSON."""

    fuel_effect: float
    compound_offset: dict[str, float]
    deg_lin: dict[str, float]
    deg_sq: dict[str, float]
    race_base: dict[str, float]
    driver_skill: dict[str, float]
    residual_sigma: float
    training_laps: int
    feature_names: list[str] = field(default_factory=list)
    #: Measured pit-lane loss (seconds) and safety-car rate per lap.
    pit_loss_s: float = 21.0
    sc_rate_per_lap: float = DEFAULT_SC_RATE

    # ---------------------------------------------------------------- fitting
    @classmethod
    def fit(cls, races: dict[str, object]) -> DegradationModel:
        """Fit the model on a mapping of ``race_key -> RaceContext``."""
        frame = _design_frame(races)
        names, matrix, target, references = _build_design(frame)
        scales = _column_scales(matrix)
        scaled = matrix.multiply(1.0 / scales).tocsr()
        solution = lsqr(scaled, target / 1.0, damp=0.0, iter_lim=400, atol=1e-10, btol=1e-10)
        coefs = solution[0] / scales
        params = dict(zip(names, coefs, strict=True))

        residual = target - scaled @ solution[0]
        compound_offset, offset_level = _centred(
            params, DRY_COMPOUNDS, references["compound"], "compound_"
        )
        driver_skill, skill_level = _centred(
            params, _driver_levels(names, references["driver"]), references["driver"], "driver_"
        )
        model = cls(
            fuel_effect=float(params["fuel_kg"]),
            compound_offset=compound_offset,
            deg_lin={c: float(params[f"age_{c}"]) for c in DRY_COMPOUNDS},
            deg_sq={c: float(params[f"age2_{c}"]) for c in DRY_COMPOUNDS},
            race_base=_shifted(params, "race_", offset_level + skill_level),
            driver_skill=driver_skill,
            residual_sigma=float(np.sqrt(np.mean(residual**2))),
            training_laps=int(len(frame)),
            feature_names=names,
            pit_loss_s=pit.default_pit_loss(races),
            sc_rate_per_lap=pit.neutralised_share(races),
        )
        LOGGER.info(
            "fitted on %d laps: fuel %.4f s/kg, sigma %.3f s",
            model.training_laps,
            model.fuel_effect,
            model.residual_sigma,
        )
        return model

    # ------------------------------------------------------------- prediction
    def predict(
        self,
        race_key: str,
        compound: str,
        tyre_age: int | float,
        fuel_kg: float,
        driver: str | None = None,
        lap_number: int | None = None,
        race_base: float | None = None,
    ) -> float:
        """Predicted lap time in seconds for a single lap."""
        base = (
            race_base
            if race_base is not None
            else self.race_base_value(race_key)
        )
        skill = self.driver_skill.get(driver, 0.0) if driver else 0.0
        value = (
            base
            + skill
            + self.compound_offset[compound]
            + self.fuel_effect * fuel_kg
            + self.deg_lin[compound] * tyre_age
            + self.deg_sq[compound] * tyre_age**2
        )
        return float(value)

    def circuit_floor(self, race_key: str) -> float:
        """Theoretical fastest lap at this circuit: new tyres, empty track."""
        base = self.race_base_value(race_key)
        return float(
            base
            + min(self.compound_offset.values())
            + min(self.deg_lin.values()) * 2
            + min(self.deg_sq.values()) * 4
            + self.fuel_effect * START_FUEL_KG
        )

    def race_base_value(self, race_key: str) -> float:
        """The fitted intercept for ``race_key``, or the field mean if unseen.

        A circuit the model has never seen (a fresh Grand Prix) still needs a
        starting point, and the mean of the fitted intercepts is the honest
        one; use :meth:`calibrate_race_base` to replace it with a measured
        value as soon as clean laps exist. Fail loudly if nothing was fitted,
        because a mean of an empty set is not a lap time.
        """
        if race_key in self.race_base:
            return float(self.race_base[race_key])
        if not self.race_base:
            raise ValueError("model has no fitted race intercepts")
        return float(np.mean(list(self.race_base.values())))

    def predict_frame(self, frame: pd.DataFrame, race_key: str, race_base: float | None = None) -> np.ndarray:
        """Vectorised prediction over a tidy lap frame.

        A raw ``RaceContext.laps`` frame is accepted and normalised, so the
        caller does not have to know the model's column names.
        """
        frame = tidy_laps(frame)
        if frame.empty:
            return np.array([])
        base = race_base if race_base is not None else self.race_base_value(race_key)
        out = np.full(len(frame), base, dtype=float)
        out += frame["driver"].map(self.driver_skill).fillna(0.0).to_numpy()
        out += frame["compound"].map(self.compound_offset).fillna(0.0).to_numpy()
        out += self.fuel_effect * frame["fuel_kg"].to_numpy()
        age = frame["tyre_age"].to_numpy(dtype=float)
        out += frame["compound"].map(self.deg_lin).fillna(0.0).to_numpy() * age
        out += frame["compound"].map(self.deg_sq).fillna(0.0).to_numpy() * age**2
        return out

    def calibrate_race_base(self, laps: pd.DataFrame) -> float:
        """Estimate the intercept of a race the model has never seen.

        This is what you would do with a fresh Grand Prix: run one or two
        practice sessions, keep the clean green-flag laps, and solve for the
        single intercept that best explains them using the already fitted
        fuel, compound and degradation coefficients.

        Any lap frame is accepted, not only a pre-tidied one.
        """
        laps = tidy_laps(laps)
        if laps.empty:
            return float(np.mean(list(self.race_base.values())))
        work = laps.copy()
        work = work.assign(
            known=(
                work["driver"].map(self.driver_skill).fillna(0.0)
                + work["compound"].map(self.compound_offset).fillna(0.0)
                + self.fuel_effect * work["fuel_kg"]
            ).to_numpy()
        )
        age = work["tyre_age"].to_numpy(dtype=float)
        work["known"] = work["known"].to_numpy() + (
            work["compound"].map(self.deg_lin).fillna(0.0).to_numpy() * age
            + work["compound"].map(self.deg_sq).fillna(0.0).to_numpy() * age**2
        )
        return float((work[TARGET] - work["known"]).mean())

    # ---------------------------------------------------------- validation
    def cross_validate(self, races: dict[str, object], folds: int = 5) -> dict[str, float]:
        """Leave-races-out validation of the degradation and fuel terms.

        Each fold holds out a slice of races, fits a fresh model on the rest and
        then predicts the held-out races using ``calibrate_race_base`` (i.e. as
        if the circuit were new and only its clean laps were available). This
        measures the accuracy of the *shape* of the model - fuel effect and
        degradation - rather than of memorising each circuit.

        A race whose every lap was cleaned away (a storm-shortened race run
        entirely behind a safety car) has no lap to score, so it is skipped and
        reported in ``races_skipped`` rather than being averaged in as a NaN:
        one unscoreable race used to poison the whole report.
        """
        keys = sorted(races)
        buckets = [keys[i::folds] for i in range(folds)]
        race_errors: list[float] = []
        residuals: list[np.ndarray] = []
        laps_scored = 0
        for held_out in buckets:
            if not held_out:
                continue
            train = {k: v for k, v in races.items() if k not in held_out}
            if not train:
                continue  # this fold trains on nothing; its score means nothing
            fold_model = DegradationModel.fit(train)
            for key in held_out:
                tidy = tidy_laps(races[key].laps)
                if TARGET not in tidy.columns or tidy.empty:
                    continue  # nothing clean to score in this race
                truth = tidy[TARGET].to_numpy(dtype=float)
                if truth.size == 0 or not np.isfinite(truth).all():
                    continue
                base = fold_model.calibrate_race_base(tidy)
                pred = fold_model.predict_frame(tidy, key, race_base=base)
                if pred.size != truth.size:
                    continue  # a fold model that cannot see this race at all
                error = pred - truth
                race_errors.append(float(error.mean()))
                residuals.append(error)
                laps_scored += int(truth.size)

        if residuals:
            pooled = np.concatenate(residuals)
            mean_error = float(pooled.mean())
            absolute = float(np.abs(pooled).mean())
            rmse = float(np.sqrt(np.mean(pooled**2)))
        else:
            mean_error = absolute = rmse = float("nan")
        return {
            "folds": len(buckets),
            "races": len(race_errors),
            "races_skipped": len(races) - len(race_errors),
            "laps": laps_scored,
            "mean_error_s": mean_error,
            "mae_s": absolute,
            "rmse_s": rmse,
        }

    # ------------------------------------------------------------ artifacts
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "fuel_effect": self.fuel_effect,
            "compound_offset": self.compound_offset,
            "deg_lin": self.deg_lin,
            "deg_sq": self.deg_sq,
            "race_base": self.race_base,
            "driver_skill": self.driver_skill,
            "residual_sigma": self.residual_sigma,
            "training_laps": self.training_laps,
            "pit_loss_s": self.pit_loss_s,
            "sc_rate_per_lap": self.sc_rate_per_lap,
            "feature_names": self.feature_names,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> DegradationModel:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**payload)
    def summary(self) -> dict[str, object]:
        """Human-readable summary of the fitted coefficients."""
        offset = {k: round(v, 3) for k, v in sorted(self.compound_offset.items(), key=lambda kv: kv[1])}
        return {
            "training_laps": self.training_laps,
            "fuel_effect_s_per_kg": round(self.fuel_effect, 4),
            "fuel_effect_s_per_race": round(self.fuel_effect * START_FUEL_KG, 2),
            "compound_offset_s": offset,
            "degradation_s_per_lap_at_10": {
                c: round(self.deg_lin[c] * 10 + self.deg_sq[c] * 100, 4) for c in DRY_COMPOUNDS
            },
            "degradation_s_per_lap_at_20": {
                c: round(self.deg_lin[c] * 20 + self.deg_sq[c] * 400, 4) for c in DRY_COMPOUNDS
            },
            "residual_sigma_s": round(self.residual_sigma, 3),
            "pit_loss_s": round(self.pit_loss_s, 2),
            "safety_car_rate_per_lap": round(self.sc_rate_per_lap, 5),
            "races": len(self.race_base),
            "drivers": len(self.driver_skill),
        }


@dataclass
class FitReport:
    """Everything worth recording about one model fit."""

    model: DegradationModel
    in_sample_sigma_s: float
    cross_validated: dict[str, float]

    @classmethod
    def build(cls, races: dict[str, object], folds: int = 5) -> FitReport:
        model = fit_degradation(races)
        return cls(
            model=model,
            in_sample_sigma_s=model.residual_sigma,
            cross_validated=model.cross_validate(races, folds=folds),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "summary": self.model.summary(),
            "cross_validation": self.cross_validated,
        }


# --------------------------------------------------------------------- helpers
def tidy_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Rename a cleaned lap frame into the column names the model uses.

    Public because every other method takes lap frames in either shape:
    ``RaceContext.laps`` as loaded, or a frame already in model terms.
    """
    frame = laps.rename(
        columns={
            "name_acronym": "driver",
            "race_slug": "race",
        }
    )
    return frame[[c for c in ("race", "year", "driver", "compound", "tyre_age", "fuel_kg", TARGET) if c in frame.columns]]


def _design_frame(races: dict[str, object]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for key, race in races.items():
        frame = tidy_laps(race.laps)
        if frame.empty:
            continue
        frame["race"] = key
        frames.append(frame)
    if not frames:
        raise ValueError("no clean laps available to fit the model")
    return pd.concat(frames, ignore_index=True)


def _build_design(frame: pd.DataFrame) -> tuple[list[str], sparse.csr_matrix, np.ndarray, dict[str, str]]:
    """One-hot encode races and drivers, and build the numeric columns."""
    race_dummies = pd.get_dummies(frame["race"], prefix="race", dtype=float)
    driver_dummies = pd.get_dummies(frame["driver"], prefix="driver", dtype=float)
    compound_dummies = pd.get_dummies(frame["compound"], prefix="compound", dtype=float)
    compound_dummies = compound_dummies.reindex(
        columns=[f"compound_{c}" for c in DRY_COMPOUNDS], fill_value=0.0
    )
    # The race dummies already span the all-ones vector, so a global intercept would be
    # redundant. The compound dummies and the driver dummies span the same vector as
    # well: every lap has exactly one compound and one driver. Keeping those blocks
    # complete makes the design rank deficient, and lsqr then splits an arbitrary
    # constant between the blocks, which is what mis-identifies the compound offsets
    # and the driver skills. Each of the two blocks therefore drops one reference
    # level; ``_centred`` restores it as a sum-to-zero contrast and reports the
    # constant it carried, which ``_shifted`` folds into the race base so that every
    # fitted prediction stays identical.
    compound_reference = DRY_COMPOUNDS[0]
    compound_dummies = compound_dummies.drop(columns=[f"compound_{compound_reference}"])
    driver_reference = sorted(frame["driver"].unique())[0]
    driver_dummies = driver_dummies.drop(columns=[f"driver_{driver_reference}"])
    age = frame["tyre_age"].to_numpy(dtype=float)
    for compound in DRY_COMPOUNDS:
        compound_dummies[f"age_{compound}"] = age * (frame["compound"] == compound).to_numpy()
        compound_dummies[f"age2_{compound}"] = age**2 * (frame["compound"] == compound).to_numpy()
    compound_dummies["fuel_kg"] = frame["fuel_kg"].to_numpy(dtype=float)

    blocks = [race_dummies, driver_dummies, compound_dummies]
    names: list[str] = []
    matrices: list[sparse.csr_matrix] = []
    for block in blocks:
        names.extend(block.columns.tolist())
        matrices.append(sparse.csr_matrix(block.to_numpy(dtype=float)))
    matrix = sparse.hstack(matrices, format="csr")
    return names, matrix, frame[TARGET].to_numpy(dtype=float), {
        "compound": compound_reference,
        "driver": driver_reference,
    }


def _centred(
    params: dict[str, float], levels: list[str], reference: str, prefix: str
) -> tuple[dict[str, float], float]:
    """Rebuild every level of a categorical block as a sum-to-zero contrast.

    ``_build_design`` drops one level per block so the design stays full rank, which
    leaves that level's coefficient implicit: the block contributes a constant that
    no single column owns. Subtracting the mean of the fitted levels pins the
    reference level to the negative of that mean, giving all levels -- including the
    dropped one -- an offset that reads as "relative to the average level" while
    leaving the block's contribution to every lap unchanged. The returned correction
    is the reference level's offset: the constant that ``_shifted`` must remove
    from the race base so the two blocks do not absorb it twice.
    """
    fitted = {level: float(params.get(f"{prefix}{level}", 0.0)) for level in levels}
    mean = float(np.mean(list(fitted.values())))
    offsets = {level: value - mean for level, value in fitted.items()}
    return offsets, offsets[reference]


def _driver_levels(names: list[str], reference: str) -> list[str]:
    """Every driver in the frame, reference level included."""
    levels = sorted(name[len("driver_"):] for name in names if name.startswith("driver_"))
    if reference not in levels:
        levels.append(reference)
    return levels


def _shifted(params: dict[str, float], prefix: str, correction: float) -> dict[str, float]:
    """Extract a level block after removing an already-accounted constant."""
    return {
        key[len(prefix):]: value - correction
        for key, value in params.items()
        if key.startswith(prefix)
    }



def _column_scales(matrix: sparse.csr_matrix) -> np.ndarray:
    """Column norms, used to keep the least-squares solve well conditioned.

    One-hot columns have norm sqrt(n) and are left unscaled; the raw numeric
    columns (fuel, tyre age, age squared) are scaled to unit norm so that lsqr
    converges quickly and the coefficients stay comparable.
    """
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=0)).ravel())
    norms[norms == 0] = 1.0
    return norms


def fit_degradation(races: dict[str, object]) -> DegradationModel:
    """Fit a :class:`DegradationModel` on ``{race_key: RaceContext}``."""
    return DegradationModel.fit(races)


def load_model(path: str | Path) -> DegradationModel:
    return DegradationModel.load(path)

