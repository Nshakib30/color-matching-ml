"""
kubelka_munk.py
===============
Kubelka-Munk physics layer for dye characterization.
Fits K/S vs. concentration curves for each base dye.

Author: ColorMatching ML Project
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize
from typing import Optional
from .color_metrics import reflectance_to_KS, KS_to_reflectance, WAVELENGTHS


class DyeKSModel:
    """
    Represents a single base dye's K/S vs. concentration relationship.
    Fits a spline at each wavelength through calibration measurements.
    """

    def __init__(self, dye_name: str):
        self.dye_name = dye_name
        self.splines: dict = {}          # wavelength → spline
        self.concentrations: np.ndarray = None
        self.KS_matrix: np.ndarray = None  # shape (n_conc, 40)
        self.is_fitted: bool = False

    def fit(self, concentrations: list, reflectance_matrix: np.ndarray):
        """
        Fit K/S splines from calibration data.

        Parameters:
            concentrations    : list of floats, e.g. [0.0005, 0.001, 0.005, ...]
            reflectance_matrix: shape (n_conc, 40), values in [0,100]
        """
        self.concentrations = np.array(sorted(concentrations))
        sort_idx = np.argsort(concentrations)

        R = reflectance_matrix[sort_idx] / 100.0  # normalize
        R = np.clip(R, 0.001, 0.999)
        self.KS_matrix = (1 - R) ** 2 / (2 * R)

        for i, wl in enumerate(WAVELENGTHS):
            if len(self.concentrations) >= 4:
                sp = UnivariateSpline(
                    self.concentrations,
                    self.KS_matrix[:, i],
                    k=3,        # cubic spline
                    s=0,        # interpolate exactly
                    ext=3       # return boundary value outside range
                )
            else:
                # Too few points → linear interpolation
                from scipy.interpolate import interp1d
                sp = interp1d(
                    self.concentrations,
                    self.KS_matrix[:, i],
                    fill_value=(self.KS_matrix[0, i], self.KS_matrix[-1, i]),
                    bounds_error=False
                )
            self.splines[wl] = sp

        self.is_fitted = True

    def predict_KS(self, concentration: float) -> np.ndarray:
        """
        Predict K/S spectrum at a given concentration.
        Returns array of shape (40,).
        """
        if not self.is_fitted:
            raise RuntimeError(f"DyeKSModel for '{self.dye_name}' not fitted yet.")
        concentration = max(concentration, 0.0)
        return np.array([float(self.splines[wl](concentration)) for wl in WAVELENGTHS])

    def predict_reflectance(self, concentration: float) -> np.ndarray:
        """Predict reflectance spectrum at a given concentration."""
        KS = self.predict_KS(concentration)
        return KS_to_reflectance(KS)

    def get_Lab(self, concentration: float) -> np.ndarray:
        """Get L*a*b* color at a given concentration."""
        from .color_metrics import reflectance_to_Lab
        R = self.predict_reflectance(concentration)
        return reflectance_to_Lab(R)

    def __repr__(self):
        if self.is_fitted:
            return (f"DyeKSModel('{self.dye_name}', "
                    f"n_points={len(self.concentrations)}, "
                    f"range=[{self.concentrations.min():.4f}, {self.concentrations.max():.1f}])")
        return f"DyeKSModel('{self.dye_name}', not fitted)"


class SubstrateModel:
    """
    Represents the undyed substrate (polyester tape or thread).
    Provides baseline K/S for mixture calculations.
    """

    def __init__(self, name: str = "polyester"):
        self.name = name
        self.KS: Optional[np.ndarray] = None

    def fit_from_reflectance(self, reflectance: np.ndarray):
        """Set substrate K/S from a white/undyed substrate measurement."""
        R = np.asarray(reflectance)
        if R.max() > 1.5:
            R = R / 100.0
        R = np.clip(R, 0.001, 0.999)
        self.KS = (1 - R) ** 2 / (2 * R)

    def get_KS(self) -> np.ndarray:
        if self.KS is None:
            # Default: assume near-white polyester substrate
            # Approximate reflectance ~85% across visible range
            R_default = np.full(40, 0.85)
            self.KS = (1 - R_default) ** 2 / (2 * R_default)
        return self.KS


class KMDyeLibrary:
    """
    Collection of DyeKSModel objects for all 16+ base dyes.
    Main interface for the physics layer.
    """

    def __init__(self, substrate: Optional[SubstrateModel] = None):
        self.models: dict[str, DyeKSModel] = {}
        self.substrate = substrate or SubstrateModel()

    def add_dye(self, model: DyeKSModel):
        self.models[model.dye_name] = model

    def fit_from_calibration_df(self, calib_df: pd.DataFrame):
        """
        Build all dye models from the parsed calibration DataFrame.

        Parameters:
            calib_df: DataFrame with columns:
                ['dye_name', 'concentration', 'R360', ..., 'R750']
        """
        r_cols = [f"R{wl}" for wl in WAVELENGTHS]
        grouped = calib_df.groupby("dye_name")

        for dye_name, group in grouped:
            group = group.dropna(subset=["concentration"])
            group = group[group["concentration"] > 0]  # skip zero concentration

            if len(group) < 2:
                print(f"  Skipping '{dye_name}': insufficient calibration points ({len(group)})")
                continue

            concentrations = group["concentration"].values
            reflectance_matrix = group[r_cols].values

            model = DyeKSModel(dye_name)
            model.fit(concentrations, reflectance_matrix)
            self.models[dye_name] = model
            print(f"  [OK] Fitted: {model}")

    def predict_mixture(
        self,
        recipe: dict,   # {dye_name: concentration}
    ) -> np.ndarray:
        """
        Predict reflectance of a dye mixture using K-M additivity.

        Parameters:
            recipe: dict mapping dye_name → concentration (% owf)

        Returns:
            reflectance array [0,1], shape (40,)
        """
        total_KS = self.substrate.get_KS().copy()

        for dye_name, conc in recipe.items():
            if dye_name not in self.models:
                raise KeyError(f"Dye '{dye_name}' not in library. Available: {list(self.models.keys())}")
            if conc > 0:
                total_KS += self.models[dye_name].predict_KS(conc)

        return KS_to_reflectance(total_KS)

    def optimize_recipe_KM(
        self,
        target_reflectance: np.ndarray,
        dye_combo: tuple,
        conc_bounds: tuple = (0.001, 5.0),
        n_starts: int = 8,
    ) -> dict:
        """
        Optimize dye concentrations for a given dye combination
        to minimize spectral RMS error vs. target.

        Returns:
            {'recipe': {dye: conc}, 'rms_error': float, 'convergence': bool}
        """
        from .color_metrics import reflectance_to_Lab, delta_E_2000

        target_R = np.asarray(target_reflectance)
        if target_R.max() > 1.5:
            target_R = target_R / 100.0
        target_Lab = reflectance_to_Lab(target_R)

        n = len(dye_combo)
        best_dE = float("inf")
        best_x = None

        def objective(x):
            recipe = {d: max(c, 0.0) for d, c in zip(dye_combo, x)}
            try:
                pred_R = self.predict_mixture(recipe)
                pred_Lab = reflectance_to_Lab(pred_R)
                return delta_E_2000(target_Lab, pred_Lab)
            except Exception:
                return 1e6

        bounds = [conc_bounds] * n

        for _ in range(n_starts):
            x0 = np.random.uniform(conc_bounds[0], min(2.0, conc_bounds[1]), n)
            try:
                res = minimize(
                    objective, x0, method="L-BFGS-B", bounds=bounds,
                    options={"ftol": 1e-10, "maxiter": 500}
                )
                if res.fun < best_dE:
                    best_dE = res.fun
                    best_x = res.x
            except Exception:
                continue

        if best_x is None:
            return {"recipe": {}, "dE2000": float("inf"), "convergence": False}

        recipe = {d: round(float(c), 6) for d, c in zip(dye_combo, best_x)}
        return {"recipe": recipe, "dE2000": round(best_dE, 4), "convergence": True}

    @property
    def dye_names(self) -> list:
        return list(self.models.keys())

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"KMDyeLibrary saved to {path}")

    @classmethod
    def load(cls, path: str) -> "KMDyeLibrary":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"KMDyeLibrary loaded from {path}")
        return obj

    def __repr__(self):
        return f"KMDyeLibrary({len(self.models)} dyes: {self.dye_names})"

