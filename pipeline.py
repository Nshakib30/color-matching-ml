"""
pipeline.py
===========
Main orchestrator: data loading → model training → prediction.

Usage:
    python pipeline.py --substrate tape --train
    python pipeline.py --substrate tape --predict --color "E001"
    python pipeline.py --substrate tape --predict --lab 45.2 12.1 -8.3
    python pipeline.py --substrate tape --benchmark

Author: ColorMatching ML Project
"""

import argparse
import json
import os
import sys

# Force UTF-8 output on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# -- Add project root to path ------------------------------------------
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.parsing.qtx_parser import (
    parse_calibration_qtx, parse_standard_qtx, reflectance_columns
)
from src.physics.kubelka_munk import KMDyeLibrary, SubstrateModel
from src.physics.color_metrics import (
    reflectance_to_Lab, delta_E_2000, batch_reflectance_to_Lab, WAVELENGTHS
)
from src.models.ml_models import (
    ForwardColorModel, InverseColorModel, ModelBenchmark,
    build_recipe_features, build_target_features
)
from src.search.combo_search import ComboSearchEngine, format_recipe_output


R_COLS = [f"R{wl}" for wl in WAVELENGTHS]

# --- Paths ------------------------------------------------------------
DATA_DIR = Path("G:/My Drive/ColorMission Backup QTX files")
PROC_DIR = Path("C:/ColorMatchingML/data/processed")
OUT_DIR  = Path("C:/ColorMatchingML/outputs")
PROC_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- File paths by substrate ------------------------------------------
SUBSTRATE_FILES = {
    "tape": {
        "calibration": DATA_DIR / "FZL TAPE DYES Calibration.qtx",
        "dyes":        DATA_DIR / "FZL TAPE DYES.qtx",
        "recipes":     DATA_DIR / "tape_recipes_final.csv",
        "color_book":  DATA_DIR / "FZL Tape COLOR BOOK.qtx",
        "standards":   DATA_DIR / "FZL STD COLOR.qtx",
    },
    "thread": {
        "calibration": DATA_DIR / "FZL THREAD DYES Calibration.qtx",
        "dyes":        DATA_DIR / "FZL THREAD DYES.qtx",
        "recipes":     DATA_DIR / "thread_recipes_final.csv",
        "color_book1": DATA_DIR / "FZL Thread color book 1.qtx",
        "color_book2": DATA_DIR / "FZL Thread color book 2.qtx",
        "standards":   DATA_DIR / "FZL STD COLOR.qtx",
    },
}


# =======================================================================
# STEP 1 — Data Loading
# =======================================================================

def load_recipe_data(substrate: str) -> tuple:
    """
    Load and merge recipe CSV with corresponding QTX color measurements.

    Returns:
        recipe_df  : DataFrame [Standard, dye_cols...]
        reflect_df : DataFrame [Standard, R360..R750]
        dye_columns: list of dye name strings
    """
    files = SUBSTRATE_FILES[substrate]
    print(f"\n[1] Loading {substrate} recipe data...")

    # -- Load recipe CSV ----------------------------------------------
    recipe_df = pd.read_csv(files["recipes"])
    print(f"    Recipes loaded: {len(recipe_df)} rows")

    # Identify dye columns (everything except Standard, DE_D65, MI_A)
    meta_cols = ["Standard", "DE_D65", "MI_A"]
    dye_columns = [c for c in recipe_df.columns if c not in meta_cols]
    print(f"    Dyes in recipe: {len(dye_columns)} → {dye_columns}")

    # Fill NaN concentrations with 0
    recipe_df[dye_columns] = recipe_df[dye_columns].fillna(0.0)

    # -- Load color book QTX for reflectance measurements -------------
    if substrate == "tape":
        std_df = parse_standard_qtx(str(files["color_book"]))
    else:
        std_df1 = parse_standard_qtx(str(files["color_book1"]))
        std_df2 = parse_standard_qtx(str(files["color_book2"]))
        std_df  = pd.concat([std_df1, std_df2], ignore_index=True)

    std_df = std_df.rename(columns={"name": "Standard"})
    print(f"    Color book spectra: {len(std_df)} entries")

    # -- Merge on Standard name -------------------------------------
    # Normalize names for matching
    recipe_df["_std_key"] = recipe_df["Standard"].astype(str).str.strip().str.upper()
    std_df["_std_key"]    = std_df["Standard"].astype(str).str.strip().str.upper()

    merged = recipe_df.merge(
        std_df[["_std_key"] + R_COLS],
        on="_std_key",
        how="inner"
    )
    merged.drop(columns=["_std_key"], inplace=True)

    print(f"    Matched records: {len(merged)} (of {len(recipe_df)} recipes)")

    reflect_df = merged[["Standard"] + R_COLS]
    final_recipe_df = merged[["Standard"] + dye_columns]

    return final_recipe_df, reflect_df, dye_columns


# =======================================================================
# STEP 2 — K-M Library
# =======================================================================

def build_km_library(substrate: str) -> KMDyeLibrary:
    """Parse calibration QTX and fit K/S models for all dyes."""
    files = SUBSTRATE_FILES[substrate]
    km_path = PROC_DIR / f"km_library_{substrate}.pkl"

    if km_path.exists():
        print(f"\n[2] Loading cached K-M library ({substrate})...")
        return KMDyeLibrary.load(str(km_path))

    print(f"\n[2] Building K-M library for {substrate}...")
    calib_df = parse_calibration_qtx(str(files["calibration"]))
    print(f"    Calibration records: {len(calib_df)}")
    print(f"    Unique dyes: {calib_df['dye_name'].nunique()}")

    km = KMDyeLibrary()
    km.fit_from_calibration_df(calib_df)
    km.save(str(km_path))

    return km


# =======================================================================
# STEP 3 — ML Model Training
# =======================================================================

def train_all_models(
    substrate: str,
    recipe_df: pd.DataFrame,
    reflect_df: pd.DataFrame,
    dye_columns: list,
):
    """Train forward and inverse models for all model types."""
    print(f"\n[3] Training ML models for {substrate}...")

    # Build features
    X_forward = build_recipe_features(recipe_df, dye_columns, substrate)
    y_forward  = reflect_df[R_COLS].values

    print(f"    Feature matrix: {X_forward.shape}")
    print(f"    Target spectra: {y_forward.shape}")

    # -- Convert reflectances to Lab for inverse model ----------------
    print("    Converting reflectance → L*a*b*...")
    Labs = batch_reflectance_to_Lab(y_forward)
    X_inverse = build_target_features(Labs)
    y_inverse  = recipe_df[dye_columns].fillna(0.0).values

    print(f"    Inverse feature matrix: {X_inverse.shape}")
    print(f"    Inverse targets: {y_inverse.shape}")

    models_to_train = ["random_forest", "extra_trees", "mlp"]
    if __import__("importlib").util.find_spec("xgboost"):
        models_to_train.append("xgboost")
    if __import__("importlib").util.find_spec("lightgbm"):
        models_to_train.append("lightgbm")

    trained_models = {}

    for model_name in models_to_train:
        print(f"\n  -- {model_name.upper()} --")

        # Forward model
        fwd = ForwardColorModel(model_name)
        fwd.dye_columns   = dye_columns
        fwd.substrate_type = substrate
        fwd.train(X_forward, y_forward)
        fwd_path = str(PROC_DIR / f"forward_{model_name}_{substrate}.pkl")
        fwd.save(fwd_path)

        # Inverse model
        inv = InverseColorModel(model_name)
        inv.dye_columns = dye_columns
        inv.train(X_inverse, y_inverse)
        inv_path = str(PROC_DIR / f"inverse_{model_name}_{substrate}.pkl")
        inv.save(inv_path)

        trained_models[model_name] = {"forward": fwd, "inverse": inv}

    print(f"\n  [OK] All models saved to {PROC_DIR}")
    return trained_models, X_forward, y_forward, X_inverse, y_inverse


# =======================================================================
# STEP 4 — Benchmark
# =======================================================================

def run_benchmark(
    X_forward, y_forward, X_inverse, y_inverse, substrate: str
):
    """Cross-validate all models and print comparison table."""
    print(f"\n[4] Benchmarking all models ({substrate})...")

    bench = ModelBenchmark(cv_folds=5)

    print("\n  --- FORWARD MODEL (recipe → reflectance) ---")
    fwd_results = bench.run_forward_benchmark(X_forward, y_forward)

    print("\n  --- INVERSE MODEL (Lab → concentrations) ---")
    inv_results = bench.run_inverse_benchmark(X_inverse, y_inverse)

    bench.print_summary()

    # Save benchmark results
    fwd_results.to_csv(str(OUT_DIR / f"benchmark_forward_{substrate}.csv"), index=False)
    inv_results.to_csv(str(OUT_DIR / f"benchmark_inverse_{substrate}.csv"), index=False)
    print(f"  Benchmark results saved to {OUT_DIR}")

    return bench


# =======================================================================
# STEP 5 — Prediction Interface
# =======================================================================

class ColorMatchingPredictor:
    """
    High-level prediction interface.
    Given a color name or Lab values, returns the best dye recipe.
    """

    def __init__(
        self,
        substrate: str,
        model_name: str = "random_forest",
        use_km_search: bool = True,
    ):
        self.substrate = substrate
        self.model_name = model_name
        self.use_km_search = use_km_search

        # Load models
        print(f"\nLoading ColorMatchingPredictor ({substrate}, {model_name})...")

        self.km = KMDyeLibrary.load(str(PROC_DIR / f"km_library_{substrate}.pkl"))

        inv_path = PROC_DIR / f"inverse_{model_name}_{substrate}.pkl"
        if inv_path.exists():
            self.inverse_model = InverseColorModel.load(str(inv_path))
        else:
            self.inverse_model = None
            print(f"  Warning: Inverse model not found at {inv_path}")

        # Load standard color library for name-based lookup
        self.std_library = parse_standard_qtx(
            str(SUBSTRATE_FILES[substrate]["standards"])
        )
        # Also load color books
        if substrate == "tape":
            cb = parse_standard_qtx(str(SUBSTRATE_FILES["tape"]["color_book"]))
        else:
            cb1 = parse_standard_qtx(str(SUBSTRATE_FILES["thread"]["color_book1"]))
            cb2 = parse_standard_qtx(str(SUBSTRATE_FILES["thread"]["color_book2"]))
            cb  = pd.concat([cb1, cb2], ignore_index=True)
        self.color_book = cb

        if use_km_search:
            self.search_engine = ComboSearchEngine(self.km, dE_threshold=1.0)
        print("  [OK] Ready.")

    def predict_from_name(self, color_name: str, top_n: int = 3) -> str:
        """
        Predict recipe for a named color (e.g., 'E001', 'PANTONE 18-4051 TCX').

        Looks up target reflectance in the color books, then runs search.
        """
        # Try color book first
        df = self.color_book.copy()
        df["_key"] = df["name"].str.strip().str.upper()
        match = df[df["_key"] == color_name.strip().upper()]

        if match.empty:
            # Try standards file
            df2 = self.std_library.copy()
            df2["_key"] = df2["name"].str.strip().str.upper()
            match = df2[df2["_key"] == color_name.strip().upper()]

        if match.empty:
            return f"Color '{color_name}' not found in library.\nAvailable colors: {list(df['name'])[:10]}..."

        target_R = match.iloc[0][R_COLS].values.astype(float)
        return self._run_prediction(target_R, top_n)

    def predict_from_Lab(self, L: float, a: float, b: float, top_n: int = 3) -> str:
        """
        Predict recipe from L*a*b* values.
        Uses ML inverse model as initial estimate, then K-M optimization.
        """
        target_Lab = np.array([L, a, b])
        output_lines = []
        output_lines.append(f"\n  Input: L*={L:.2f}, a*={a:.2f}, b*={b:.2f}")

        # -- ML inverse prediction (fast, approximate) -----------------
        if self.inverse_model:
            ml_recipe = self.inverse_model.predict_recipe(target_Lab)
            active = {d: c for d, c in ml_recipe.items() if c > 0.001}
            output_lines.append(f"\n  [ML Inverse Estimate — {self.model_name}]")
            output_lines.append(f"  {'Dye':<25} {'Concentration (% owf)':>22}")
            output_lines.append(f"  {'-'*49}")
            for dye, conc in sorted(active.items(), key=lambda x: -x[1]):
                output_lines.append(f"  {dye:<25} {conc:>20.4f}%")

        # -- K-M combinatorial search (precise) ------------------------
        if self.use_km_search:
            # Use best matching dye's spectrum as proxy target
            dye_labs = {
                d: reflectance_to_Lab(m.predict_reflectance(1.0))
                for d, m in self.km.models.items()
            }
            best_proxy_dye = min(dye_labs, key=lambda d: delta_E_2000(target_Lab, dye_labs[d]))
            proxy_R = self.km.models[best_proxy_dye].predict_reflectance(1.0)
            results = self.search_engine.search(proxy_R, n_best=top_n)
            output_lines.append(format_recipe_output(results, top_n))

        return "\n".join(output_lines)

    def predict_from_reflectance(self, reflectance: np.ndarray, top_n: int = 3) -> str:
        """Predict recipe from a full reflectance spectrum."""
        return self._run_prediction(reflectance, top_n)

    def _run_prediction(self, target_R: np.ndarray, top_n: int) -> str:
        """Internal: run K-M search on a target reflectance."""
        if self.use_km_search:
            results = self.search_engine.search(target_R, n_best=top_n)
            return format_recipe_output(results, top_n)
        else:
            target_R_norm = target_R / 100.0 if target_R.max() > 1.5 else target_R
            target_Lab = reflectance_to_Lab(target_R_norm)
            if self.inverse_model:
                recipe = self.inverse_model.predict_recipe(target_Lab)
                active = {d: c for d, c in recipe.items() if c > 0.001}
                lines = ["\n  [ML Inverse Prediction]"]
                for d, c in sorted(active.items(), key=lambda x: -x[1]):
                    lines.append(f"  {d:<25} {c:>20.4f}%")
                return "\n".join(lines)
            return "No prediction method available."

    def list_available_colors(self) -> list:
        """List all color names in the library."""
        return self.color_book["name"].tolist()


# =======================================================================
# CLI Entry Point
# =======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Color Matching ML System — FZL Polyester Tape & Thread"
    )
    parser.add_argument("--substrate", choices=["tape", "thread"], default="tape",
                        help="Substrate type: tape or thread")
    parser.add_argument("--train",     action="store_true", help="Train all models")
    parser.add_argument("--benchmark", action="store_true", help="Run model benchmark")
    parser.add_argument("--predict",   action="store_true", help="Run prediction")
    parser.add_argument("--color",     type=str,  default=None, help="Color name (e.g. E001)")
    parser.add_argument("--lab",       nargs=3, type=float, default=None,
                        metavar=("L", "a", "b"), help="Target L*a*b* values")
    parser.add_argument("--model",     default="random_forest",
                        choices=["random_forest", "extra_trees", "xgboost",
                                 "lightgbm", "gradient_boosting", "mlp"],
                        help="ML model to use")
    parser.add_argument("--top",       type=int, default=3, help="Number of recipes to return")
    parser.add_argument("--no-km",     action="store_true",
                        help="Use ML inverse only (skip K-M combinatorial search)")

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  COLOR MATCHING ML SYSTEM — FZL Polyester {args.substrate.upper()}")
    print(f"{'='*60}")

    if args.train or args.benchmark:
        recipe_df, reflect_df, dye_cols = load_recipe_data(args.substrate)
        km = build_km_library(args.substrate)

        trained, X_fwd, y_fwd, X_inv, y_inv = train_all_models(
            args.substrate, recipe_df, reflect_df, dye_cols
        )

        if args.benchmark:
            run_benchmark(X_fwd, y_fwd, X_inv, y_inv, args.substrate)

    if args.predict:
        predictor = ColorMatchingPredictor(
            substrate=args.substrate,
            model_name=args.model,
            use_km_search=not args.no_km,
        )

        if args.color:
            print(f"\n  Looking up: '{args.color}'")
            result = predictor.predict_from_name(args.color, top_n=args.top)
            print(result)

        elif args.lab:
            L, a, b = args.lab
            result = predictor.predict_from_Lab(L, a, b, top_n=args.top)
            print(result)

        else:
            print("\n  Error: Provide --color NAME or --lab L a b")
            sys.exit(1)


if __name__ == "__main__":
    main()

