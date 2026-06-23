"""
ml_models.py
============
Machine Learning models for color recipe prediction.
Implements:
  1. Forward Model  : [dye_ids (OHE) + concentrations] → reflectance R(λ)
  2. Inverse Model  : [target Lab + substrate] → dye concentrations
  3. Residual Model : K-M prediction + ML correction residual

Models trained: Random Forest, XGBoost, LightGBM, MLP, Gradient Boosting
Evaluation metric: ΔE₂₀₀₀

Author: ColorMatching ML Project
"""

import numpy as np
import pandas as pd
import pickle
import warnings
from pathlib import Path
from typing import Optional

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
import joblib

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not installed. Run: pip install xgboost")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    print("LightGBM not installed. Run: pip install lightgbm")

WAVELENGTHS = np.arange(360, 760, 10)
R_COLS = [f"R{wl}" for wl in WAVELENGTHS]


# =======================================================================
# Feature Engineering
# =======================================================================

def build_recipe_features(
    recipe_df: pd.DataFrame,
    dye_columns: list,
    substrate_type: str = "tape",
) -> np.ndarray:
    """
    Build feature matrix from recipe DataFrame.

    Features:
        - Dye concentrations (direct, for each of 16+ dyes)
        - Binary indicators: which dyes are active (>0)
        - Number of dyes used
        - Total dye load (sum of all concentrations)
        - Substrate encoding (0=tape, 1=thread)

    Parameters:
        recipe_df   : DataFrame with dye columns (concentration values, NaN=0)
        dye_columns : list of dye name columns
        substrate_type: 'tape' or 'thread'

    Returns:
        Feature matrix, shape (n_samples, n_features)
    """
    df = recipe_df[dye_columns].fillna(0.0).copy()

    # Raw concentrations
    X_conc = df.values  # (n, n_dyes)

    # Binary presence indicators
    X_binary = (X_conc > 0).astype(float)

    # Aggregate features
    n_dyes_used = X_binary.sum(axis=1, keepdims=True)
    total_load = X_conc.sum(axis=1, keepdims=True)

    # Substrate encoding
    sub_enc = np.full((len(df), 1), 0.0 if substrate_type == "tape" else 1.0)

    X = np.hstack([X_conc, X_binary, n_dyes_used, total_load, sub_enc])
    return X


def build_target_features(
    target_Lab: np.ndarray,
) -> np.ndarray:
    """
    Build feature vector from target L*a*b* values for inverse model.
    Adds derived features: chroma, hue angle.

    Parameters:
        target_Lab: array of shape (3,) or (N, 3)

    Returns:
        Feature array (N, 7) or (7,)
    """
    Lab = np.atleast_2d(target_Lab).astype(float)
    L, a, b = Lab[:, 0], Lab[:, 1], Lab[:, 2]

    chroma = np.sqrt(a**2 + b**2)
    hue = np.arctan2(b, a)
    hue_sin = np.sin(hue)
    hue_cos = np.cos(hue)

    X = np.column_stack([L, a, b, chroma, hue, hue_sin, hue_cos])
    return X.squeeze() if target_Lab.ndim == 1 else X


# =======================================================================
# Model Definitions
# =======================================================================

def get_model_zoo(task: str = "forward") -> dict:
    """
    Returns a dictionary of all available models.

    Parameters:
        task: 'forward' (recipe→reflectance) or 'inverse' (Lab→concentrations)
    """
    models = {
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=42,
        ),
        "gradient_boosting": MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                random_state=42,
            ),
            n_jobs=-1,
        ),
        "mlp": Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=(256, 128, 64),
                activation="relu",
                solver="adam",
                max_iter=1000,
                learning_rate_init=0.001,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
            )),
        ]),
    }

    if XGB_AVAILABLE:
        models["xgboost"] = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42,
            verbosity=0,
        ) if task == "inverse" else MultiOutputRegressor(
            xgb.XGBRegressor(
                n_estimators=300,
                learning_rate=0.08,
                max_depth=6,
                subsample=0.8,
                n_jobs=-1,
                random_state=42,
                verbosity=0,
            ),
            n_jobs=4,
        )

    if LGB_AVAILABLE:
        models["lightgbm"] = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        ) if task == "inverse" else MultiOutputRegressor(
            lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.08,
                num_leaves=31,
                n_jobs=-1,
                random_state=42,
                verbose=-1,
            ),
            n_jobs=4,
        )

    return models


# =======================================================================
# Forward Model: Recipe → Reflectance
# =======================================================================

class ForwardColorModel:
    """
    Predicts reflectance spectrum R(λ) from a dye recipe.
    Used for: 'given this recipe, what color will it produce?'
    Also used internally in the residual correction pipeline.
    """

    def __init__(self, model_name: str = "random_forest"):
        self.model_name = model_name
        self.model = None
        self.dye_columns = None
        self.substrate_type = None
        self.is_trained = False
        self.train_metrics = {}

    def prepare_data(
        self,
        recipe_df: pd.DataFrame,
        dye_columns: list,
        reflectance_df: pd.DataFrame,
        substrate_type: str = "tape",
    ):
        """
        Prepare X, y arrays for training.

        Parameters:
            recipe_df     : DataFrame with recipe info (dye concentrations)
            dye_columns   : columns for dye concentrations
            reflectance_df: DataFrame with R360..R750 columns (target reflectance)
            substrate_type: 'tape' or 'thread'

        Returns:
            X (n_samples, n_features), y (n_samples, 40)
        """
        self.dye_columns = dye_columns
        self.substrate_type = substrate_type

        X = build_recipe_features(recipe_df, dye_columns, substrate_type)
        y = reflectance_df[R_COLS].values

        return X, y

    def train(self, X: np.ndarray, y: np.ndarray, model_name: Optional[str] = None):
        """Train the forward model."""
        if model_name:
            self.model_name = model_name

        zoo = get_model_zoo("forward")
        if self.model_name not in zoo:
            raise ValueError(f"Unknown model '{self.model_name}'. Available: {list(zoo.keys())}")

        self.model = zoo[self.model_name]
        print(f"  Training {self.model_name} forward model...")
        self.model.fit(X, y)
        self.is_trained = True

        # Training MSE
        y_pred = self.model.predict(X)
        mse = mean_squared_error(y, y_pred)
        self.train_metrics = {"train_mse": mse, "train_rmse": np.sqrt(mse)}
        print(f"  [OK] Training RMSE: {self.train_metrics['train_rmse']:.4f}")

    def predict(self, recipe_features: np.ndarray) -> np.ndarray:
        """Predict reflectance from recipe features."""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet.")
        pred = self.model.predict(np.atleast_2d(recipe_features))
        return np.clip(pred, 0.0, 100.0)

    def save(self, path: str):
        joblib.dump(self, path)
        print(f"ForwardColorModel saved to {path}")

    @classmethod
    def load(cls, path: str) -> "ForwardColorModel":
        return joblib.load(path)


# =======================================================================
# Inverse Model: Target Lab → Concentrations
# =======================================================================

class InverseColorModel:
    """
    Predicts dye concentrations directly from target L*a*b* values.
    One model per dye concentration output.

    This is trained on: Lab features → [c_dye1, c_dye2, ..., c_dye16]
    """

    def __init__(self, model_name: str = "xgboost"):
        self.model_name = model_name
        self.model = None
        self.dye_columns = None
        self.is_trained = False
        self.train_metrics = {}

    def prepare_data(
        self,
        recipe_df: pd.DataFrame,
        dye_columns: list,
        reflectance_df: pd.DataFrame,
        illuminant: str = "D65",
    ):
        """
        Convert reflectance to Lab features, prepare for training.
        """
        from ..physics.color_metrics import reflectance_to_Lab

        self.dye_columns = dye_columns

        # Convert all spectra to Lab
        print("  Converting spectra to L*a*b*...")
        Labs = []
        for _, row in reflectance_df.iterrows():
            R = row[R_COLS].values.astype(float)
            Lab = reflectance_to_Lab(R, illuminant)
            Labs.append(Lab)
        Labs = np.array(Labs)

        X = build_target_features(Labs)
        y = recipe_df[dye_columns].fillna(0.0).values

        return X, y

    def train(self, X: np.ndarray, y: np.ndarray, model_name: Optional[str] = None):
        """Train the inverse model."""
        if model_name:
            self.model_name = model_name

        zoo = get_model_zoo("inverse")
        if self.model_name not in zoo:
            raise ValueError(f"Unknown model '{self.model_name}'. Available: {list(zoo.keys())}")

        base_model = zoo[self.model_name]

        # Wrap in MultiOutput if needed
        if hasattr(base_model, "predict") and not isinstance(base_model, MultiOutputRegressor):
            self.model = MultiOutputRegressor(base_model, n_jobs=-1)
        else:
            self.model = base_model

        print(f"  Training {self.model_name} inverse model ({y.shape[1]} outputs)...")
        self.model.fit(X, y)
        self.is_trained = True

        y_pred = np.clip(self.model.predict(X), 0.0, None)
        mse = mean_squared_error(y, y_pred)
        self.train_metrics = {"train_mse": mse, "train_rmse": np.sqrt(mse)}
        print(f"  [OK] Training RMSE: {self.train_metrics['train_rmse']:.6f}")

    def predict_recipe(self, target_Lab: np.ndarray) -> dict:
        """
        Predict concentrations from target L*a*b*.

        Returns:
            dict mapping dye_name → concentration (non-negative)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet.")

        X = build_target_features(np.atleast_1d(target_Lab)).reshape(1, -1)
        concs = self.model.predict(X)[0]
        concs = np.clip(concs, 0.0, None)  # no negative concentrations

        return {
            dye: round(float(c), 6)
            for dye, c in zip(self.dye_columns, concs)
        }

    def save(self, path: str):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "InverseColorModel":
        return joblib.load(path)


# =======================================================================
# Model Benchmark Suite
# =======================================================================

class ModelBenchmark:
    """
    Train and compare all available models on the same dataset.
    Reports cross-validated metrics for both forward and inverse tasks.
    """

    def __init__(self, cv_folds: int = 5):
        self.cv_folds = cv_folds
        self.results: dict = {}

    def run_forward_benchmark(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Compare all models on the forward prediction task."""
        zoo = get_model_zoo("forward")
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        rows = []

        for name, model in zoo.items():
            print(f"  Benchmarking {name}...")
            fold_rmse = []
            fold_dE = []

            for train_idx, val_idx in kf.split(X):
                X_tr, X_val = X[train_idx], X[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]

                try:
                    model.fit(X_tr, y_tr)
                    y_pred = np.clip(model.predict(X_val), 0, 100)
                    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                    fold_rmse.append(rmse)
                except Exception as e:
                    print(f"    Error: {e}")
                    fold_rmse.append(np.nan)

            row = {
                "model": name,
                "cv_rmse_mean": np.nanmean(fold_rmse),
                "cv_rmse_std": np.nanstd(fold_rmse),
                "task": "forward",
            }
            rows.append(row)
            if verbose:
                print(f"    RMSE: {row['cv_rmse_mean']:.4f} ± {row['cv_rmse_std']:.4f}")

        df = pd.DataFrame(rows).sort_values("cv_rmse_mean")
        self.results["forward"] = df
        return df

    def run_inverse_benchmark(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Compare all models on the inverse prediction task."""
        zoo = get_model_zoo("inverse")
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        rows = []

        for name, base_model in zoo.items():
            print(f"  Benchmarking {name}...")

            if not isinstance(base_model, MultiOutputRegressor):
                model = MultiOutputRegressor(base_model, n_jobs=-1)
            else:
                model = base_model

            fold_rmse = []
            for train_idx, val_idx in kf.split(X):
                X_tr, X_val = X[train_idx], X[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]
                try:
                    model.fit(X_tr, y_tr)
                    y_pred = np.clip(model.predict(X_val), 0, None)
                    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                    fold_rmse.append(rmse)
                except Exception as e:
                    print(f"    Error: {e}")
                    fold_rmse.append(np.nan)

            row = {
                "model": name,
                "cv_rmse_mean": np.nanmean(fold_rmse),
                "cv_rmse_std":  np.nanstd(fold_rmse),
                "task": "inverse",
            }
            rows.append(row)
            if verbose:
                print(f"    RMSE: {row['cv_rmse_mean']:.6f} ± {row['cv_rmse_std']:.6f}")

        df = pd.DataFrame(rows).sort_values("cv_rmse_mean")
        self.results["inverse"] = df
        return df

    def print_summary(self):
        for task, df in self.results.items():
            print(f"\n{'='*55}")
            print(f"  {task.upper()} MODEL BENCHMARK ({self.cv_folds}-fold CV)")
            print(f"{'='*55}")
            print(df.to_string(index=False))
        print()

