"""
combo_search.py
===============
Combinatorial recipe search engine.
Enforces the constraint: primarily 3-dye recipes, fallback to 2 or 4.

For 16 dyes: C(16,3) = 560 three-dye combinations.

Author: ColorMatching ML Project
"""

import numpy as np
import pandas as pd
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional
import time

from ..physics.color_metrics import (
    reflectance_to_Lab, delta_E_2000, delta_E_76, WAVELENGTHS
)


R_COLS = [f"R{wl}" for wl in WAVELENGTHS]

# Default ΔE₂₀₀₀ acceptance threshold
DEFAULT_DE_THRESHOLD = 1.0


# =======================================================================
# Pre-screening: Reduce 560 → ~50 most relevant combos
# =======================================================================

def compute_dye_Lab_at_std(km_library, std_conc: float = 0.5) -> dict:
    """
    Compute Lab value of each base dye at a standard concentration.
    Used for hue-based pre-screening.

    Returns: {dye_name: np.array([L, a, b])}
    """
    dye_labs = {}
    for dye_name, model in km_library.models.items():
        try:
            R = model.predict_reflectance(std_conc)
            Lab = reflectance_to_Lab(R)
            dye_labs[dye_name] = Lab
        except Exception:
            pass
    return dye_labs


def prescreen_combos(
    target_Lab: np.ndarray,
    dye_labs: dict,
    n_dyes: int = 3,
    top_k: int = 8,
) -> list:
    """
    Pre-filter dye combinations by hue proximity.
    Selects top_k dyes most similar in hue to the target,
    then generates all C(top_k, n_dyes) combinations.

    Reduces: C(16,3)=560 → C(8,3)=56 combinations (90% reduction).
    """
    target_hue = np.arctan2(target_Lab[2], target_Lab[1])
    target_L = target_Lab[0]

    scores = {}
    for dye_name, lab in dye_labs.items():
        dye_hue = np.arctan2(lab[2], lab[1])
        hue_diff = abs(target_hue - dye_hue)
        hue_diff = min(hue_diff, 2 * np.pi - hue_diff)  # circular distance
        L_diff = abs(target_L - lab[0]) / 100.0
        scores[dye_name] = hue_diff + 0.3 * L_diff

    # Sort by score (lower = more similar hue)
    ranked = sorted(scores, key=scores.get)
    top_dyes = ranked[:top_k]

    return list(combinations(top_dyes, n_dyes))


# =======================================================================
# Single-Combo Optimizer (called per combo)
# =======================================================================

def _optimize_single_combo(args):
    """
    Worker function for parallel combo optimization.
    Avoids pickling issues with class methods.
    """
    combo, target_Lab, km_library_path, conc_bounds, n_starts = args

    # Load library in worker process
    from ..physics.kubelka_munk import KMDyeLibrary
    km = KMDyeLibrary.load(km_library_path)

    result = km.optimize_recipe_KM(
        target_reflectance=None,  # We pass Lab directly
        dye_combo=combo,
        conc_bounds=conc_bounds,
        n_starts=n_starts,
    )
    result["combo"] = combo
    return result


# =======================================================================
# Main Search Engine
# =======================================================================

class ComboSearchEngine:
    """
    Combinatorial dye recipe search engine.

    Search priority:
      1. 3-dye combos (C(16,3) = 560) — PRIMARY
      2. 2-dye combos (C(16,2) = 120) — FALLBACK if best ΔE > threshold
      3. 4-dye combos (C(16,4) = 1820) — LAST RESORT

    Features:
      - Hue-based pre-screening (90% combo reduction)
      - scipy.optimize L-BFGS-B per combo
      - Multi-start optimization to escape local minima
    """

    def __init__(
        self,
        km_library,
        dE_threshold: float = DEFAULT_DE_THRESHOLD,
        use_prescreening: bool = True,
        prescreening_top_k: int = 9,
    ):
        self.km = km_library
        self.dE_threshold = dE_threshold
        self.use_prescreening = use_prescreening
        self.prescreening_top_k = prescreening_top_k

        # Precompute dye Lab values for pre-screening
        self._dye_labs = compute_dye_Lab_at_std(km_library)

    def search(
        self,
        target_reflectance: np.ndarray,
        n_best: int = 5,
        conc_bounds: tuple = (0.001, 5.0),
        n_starts: int = 8,
        verbose: bool = True,
    ) -> list:
        """
        Find best dye recipes for the target color.

        Parameters:
            target_reflectance: array shape (40,) in [0-100]
            n_best            : return top-N recipes
            conc_bounds       : (min, max) concentration per dye
            n_starts          : random restarts per combo optimization
            verbose           : print progress

        Returns:
            List of result dicts sorted by ΔE₂₀₀₀, length ≤ n_best
        """
        target_R = np.asarray(target_reflectance, dtype=float)
        if target_R.max() > 1.5:
            target_R_norm = target_R / 100.0
        else:
            target_R_norm = target_R.copy()
        target_Lab = reflectance_to_Lab(target_R_norm)

        all_results = []
        t0 = time.time()

        # -- Phase 1: 3-dye search --------------------------------------
        if verbose:
            print(f"\n{'-'*55}")
            print(f"  Target Lab: L={target_Lab[0]:.1f}, a={target_Lab[1]:.1f}, b={target_Lab[2]:.1f}")

        combos_3 = self._get_combos(target_Lab, n_dyes=3)
        if verbose:
            print(f"  Searching {len(combos_3)} three-dye combinations...")

        results_3 = self._run_search(
            combos_3, target_R_norm, target_Lab, conc_bounds, n_starts
        )
        all_results.extend(results_3)

        best_3_dE = min(r["dE2000"] for r in results_3) if results_3 else float("inf")
        if verbose:
            print(f"  Best 3-dye ΔE₂₀₀₀: {best_3_dE:.4f}")

        # -- Phase 2: 2-dye fallback ------------------------------------
        if best_3_dE > self.dE_threshold:
            combos_2 = self._get_combos(target_Lab, n_dyes=2)
            if verbose:
                print(f"  Threshold not met. Trying {len(combos_2)} two-dye combos...")
            results_2 = self._run_search(
                combos_2, target_R_norm, target_Lab, conc_bounds, n_starts
            )
            all_results.extend(results_2)
            best_2_dE = min(r["dE2000"] for r in results_2) if results_2 else float("inf")
            if verbose:
                print(f"  Best 2-dye ΔE₂₀₀₀: {best_2_dE:.4f}")

        # -- Phase 3: 4-dye last resort ---------------------------------
        current_best = min(r["dE2000"] for r in all_results) if all_results else float("inf")
        if current_best > self.dE_threshold:
            combos_4 = self._get_combos(target_Lab, n_dyes=4)
            if verbose:
                print(f"  Still above threshold. Trying {len(combos_4)} four-dye combos...")
            results_4 = self._run_search(
                combos_4, target_R_norm, target_Lab, conc_bounds, n_starts
            )
            all_results.extend(results_4)

        # Sort and return top-N
        all_results.sort(key=lambda x: x["dE2000"])
        top = all_results[:n_best]

        elapsed = time.time() - t0
        if verbose:
            print(f"\n  [OK] Search completed in {elapsed:.1f}s")
            print(f"  Best ΔE₂₀₀₀: {top[0]['dE2000']:.4f} ({top[0]['grade']})")
            print(f"  Best recipe:")
            for dye, conc in top[0]["recipe"].items():
                if conc > 0.0001:
                    print(f"    {dye:20s}: {conc:.4f}% owf")

        return top

    def _get_combos(self, target_Lab: np.ndarray, n_dyes: int) -> list:
        """Get combinations to try — with optional pre-screening."""
        all_dyes = self.km.dye_names

        if self.use_prescreening and len(all_dyes) > self.prescreening_top_k:
            return prescreen_combos(
                target_Lab,
                self._dye_labs,
                n_dyes=n_dyes,
                top_k=self.prescreening_top_k,
            )
        else:
            return list(combinations(all_dyes, n_dyes))

    def _run_search(
        self,
        combos: list,
        target_R: np.ndarray,
        target_Lab: np.ndarray,
        conc_bounds: tuple,
        n_starts: int,
    ) -> list:
        """Run optimization over a list of combos."""
        from ..physics.color_metrics import delta_E_2000, reflectance_to_Lab

        results = []
        for combo in combos:
            opt = self.km.optimize_recipe_KM(
                target_reflectance=target_R,
                dye_combo=combo,
                conc_bounds=conc_bounds,
                n_starts=n_starts,
            )

            recipe = opt["recipe"]
            try:
                pred_R = self.km.predict_mixture(recipe)
                pred_Lab = reflectance_to_Lab(pred_R)
                dE = delta_E_2000(target_Lab, pred_Lab)
            except Exception:
                dE = float("inf")
                pred_Lab = np.zeros(3)

            # Grade
            if dE < 1.0:
                grade = "Excellent"
            elif dE < 2.0:
                grade = "Good"
            elif dE < 3.5:
                grade = "Acceptable"
            else:
                grade = "Reject"

            results.append({
                "recipe": recipe,
                "dE2000": round(dE, 4),
                "n_dyes": len(combo),
                "dyes_used": combo,
                "grade": grade,
                "pred_Lab": pred_Lab.tolist(),
            })

        return results

    def search_from_Lab(
        self,
        L: float, a: float, b: float,
        n_best: int = 5,
    ) -> list:
        """
        Search directly from L*a*b* values (no spectrum needed).
        Converts Lab to synthetic reflectance via KM inverse for initial guess.
        """
        # Find closest color in km library at std concentration
        dye_labs = self._dye_labs
        target_Lab = np.array([L, a, b])

        # Use hue pre-screening to pick best initial dye
        ranked = sorted(dye_labs.items(),
                        key=lambda kv: delta_E_76(target_Lab, kv[1]))
        closest_dye = ranked[0][0]

        # Get its reflectance as a proxy target for optimization
        proxy_R = self.km.models[closest_dye].predict_reflectance(1.0)

        return self.search(proxy_R, n_best=n_best)


def format_recipe_output(results: list, top_n: int = 3) -> str:
    """Format recipe results as a clean human-readable string."""
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"  COLOR MATCHING RESULTS (Top {min(top_n, len(results))})")
    lines.append(f"{'='*60}")

    for i, r in enumerate(results[:top_n]):
        lines.append(f"\n  Recipe #{i+1}  |  ΔE₂₀₀₀ = {r['dE2000']:.4f}  |  {r['grade']}")
        lines.append(f"  Number of dyes: {r['n_dyes']}")
        lines.append(f"  {'Dye':<25} {'Concentration (% owf)':>22}")
        lines.append(f"  {'-'*49}")
        for dye, conc in r["recipe"].items():
            if conc > 0.0001:
                lines.append(f"  {dye:<25} {conc:>20.4f}%")
        pred = r.get("pred_Lab", [0, 0, 0])
        lines.append(f"\n  Predicted Lab: L={pred[0]:.1f}, a={pred[1]:.1f}, b={pred[2]:.1f}")

    lines.append(f"\n{'='*60}\n")
    return "\n".join(lines)

