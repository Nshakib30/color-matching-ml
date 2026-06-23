"""
color_metrics.py
================
Color science utilities:
  - Kubelka-Munk transform
  - Reflectance → CIE XYZ → CIE L*a*b*
  - ΔE₂₀₀₀ (CIEDE2000) computation
  - Metamerism Index

Author: ColorMatching ML Project
"""

import numpy as np
import pandas as pd

try:
    import colour
    COLOUR_AVAILABLE = True
except ImportError:
    COLOUR_AVAILABLE = False
    print("Warning: colour-science not installed. Run: pip install colour-science")


# --- Wavelength grid --------------------------------------------------------
WAVELENGTHS = np.arange(360, 760, 10)  # 40 points, 360-750nm


# --- Kubelka-Munk -----------------------------------------------------------

def reflectance_to_KS(R: np.ndarray) -> np.ndarray:
    """
    Convert reflectance array to Kubelka-Munk K/S ratio.
    R should be in range [0, 1] or [0, 100] — auto-normalized.

    K/S = (1 - R)² / (2R)
    """
    R = np.asarray(R, dtype=float)
    if R.max() > 1.5:
        R = R / 100.0  # normalize from % to [0,1]
    R = np.clip(R, 0.001, 0.999)
    return (1.0 - R) ** 2 / (2.0 * R)


def KS_to_reflectance(KS: np.ndarray) -> np.ndarray:
    """
    Convert K/S ratio back to reflectance [0,1].
    R = 1 + K/S - sqrt((K/S)² + 2·K/S)
    """
    KS = np.asarray(KS, dtype=float)
    KS = np.clip(KS, 0.0, None)
    return 1.0 + KS - np.sqrt(KS ** 2 + 2.0 * KS)


def mix_KS(
    dye_KS_list: list,          # list of np.ndarray, each shape (40,)
    substrate_KS: np.ndarray,   # shape (40,)
) -> np.ndarray:
    """
    Kubelka-Munk additivity: sum dye K/S contributions onto substrate.
    Returns predicted reflectance array [0,1], shape (40,).
    """
    total_KS = substrate_KS.copy().astype(float)
    for dye_KS in dye_KS_list:
        total_KS += np.asarray(dye_KS, dtype=float)
    return KS_to_reflectance(total_KS)


# --- Reflectance → Lab ------------------------------------------------------

def reflectance_to_Lab(
    reflectance: np.ndarray,
    illuminant: str = "D65",
    observer: str = "2",
) -> np.ndarray:
    """
    Convert a spectral reflectance curve to CIE L*a*b*.

    Parameters:
        reflectance: array of 40 values (360-750nm), [0-1] or [0-100]
        illuminant : 'D65', 'D50', 'A', 'C', 'F2', etc.
        observer   : '2' (1931) or '10' (1964)

    Returns:
        np.array([L*, a*, b*])
    """
    if not COLOUR_AVAILABLE:
        raise RuntimeError("colour-science required. Run: pip install colour-science")

    R = np.asarray(reflectance, dtype=float)
    if R.max() > 1.5:
        R = R / 100.0

    obs_key = "CIE 1931 2 Degree Standard Observer" if observer == "2" \
              else "CIE 1964 10 Degree Standard Observer"

    cmfs = colour.MSDS_CMFS[obs_key]
    illum_sd = colour.SDS_ILLUMINANTS[illuminant]

    sd = colour.SpectralDistribution(
        dict(zip(WAVELENGTHS, R))
    )

    XYZ = colour.sd_to_XYZ(sd, cmfs, illum_sd) / 100.0
    illum_xy = colour.CCS_ILLUMINANTS[obs_key][illuminant]
    Lab = colour.XYZ_to_Lab(XYZ, illum_xy)

    return Lab


def reflectance_to_XYZ(reflectance: np.ndarray, illuminant: str = "D65") -> np.ndarray:
    """Convert reflectance to CIE XYZ."""
    if not COLOUR_AVAILABLE:
        raise RuntimeError("colour-science required.")
    R = np.asarray(reflectance, dtype=float)
    if R.max() > 1.5:
        R = R / 100.0
    cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    illum_sd = colour.SDS_ILLUMINANTS[illuminant]
    sd = colour.SpectralDistribution(dict(zip(WAVELENGTHS, R)))
    return colour.sd_to_XYZ(sd, cmfs, illum_sd)


def batch_reflectance_to_Lab(
    reflectance_matrix: np.ndarray,  # shape (N, 40)
    illuminant: str = "D65",
) -> np.ndarray:
    """
    Convert a batch of reflectance spectra to L*a*b*.
    Returns array of shape (N, 3).
    """
    return np.array([
        reflectance_to_Lab(R, illuminant) for R in reflectance_matrix
    ])


# --- ΔE Metrics -------------------------------------------------------------

def delta_E_2000(Lab1: np.ndarray, Lab2: np.ndarray) -> float:
    """
    CIEDE2000 color difference between two L*a*b* values.
    Industry standard for color matching accuracy.
    """
    if not COLOUR_AVAILABLE:
        raise RuntimeError("colour-science required.")
    return float(colour.delta_E(Lab1, Lab2, method="CIE 2000"))


def delta_E_CMC(Lab1: np.ndarray, Lab2: np.ndarray, l: float = 2.0, c: float = 1.0) -> float:
    """
    CMC(l:c) color difference. Common in textile industry.
    Default CMC(2:1) for acceptability.
    """
    if not COLOUR_AVAILABLE:
        raise RuntimeError("colour-science required.")
    return float(colour.delta_E(Lab1, Lab2, method="CMC"))


def delta_E_76(Lab1: np.ndarray, Lab2: np.ndarray) -> float:
    """CIE 1976 ΔE*ab — simple Euclidean, useful for fast screening."""
    Lab1, Lab2 = np.asarray(Lab1), np.asarray(Lab2)
    return float(np.linalg.norm(Lab1 - Lab2))


def evaluate_recipe_accuracy(
    target_R: np.ndarray,
    predicted_R: np.ndarray,
    illuminant: str = "D65",
) -> dict:
    """
    Full accuracy report for a predicted recipe vs. target.

    Returns:
        {
          'dE2000'  : float,
          'dE_CMC'  : float,
          'dE76'    : float,
          'target_Lab'   : [L, a, b],
          'pred_Lab'     : [L, a, b],
          'grade'   : str  ('Excellent'/'Good'/'Acceptable'/'Reject')
        }
    """
    target_Lab = reflectance_to_Lab(target_R, illuminant)
    pred_Lab   = reflectance_to_Lab(predicted_R, illuminant)

    dE2000 = delta_E_2000(target_Lab, pred_Lab)
    dE_cmc = delta_E_CMC(target_Lab, pred_Lab)
    dE76   = delta_E_76(target_Lab, pred_Lab)

    if dE2000 < 1.0:
        grade = "Excellent"
    elif dE2000 < 2.0:
        grade = "Good"
    elif dE2000 < 3.5:
        grade = "Acceptable"
    else:
        grade = "Reject"

    return {
        "dE2000": round(dE2000, 4),
        "dE_CMC": round(dE_cmc, 4),
        "dE76":   round(dE76,   4),
        "target_Lab": target_Lab.tolist(),
        "pred_Lab":   pred_Lab.tolist(),
        "grade": grade,
    }


# --- Metamerism -------------------------------------------------------------

def metamerism_index(R1: np.ndarray, R2: np.ndarray) -> dict:
    """
    Compute metamerism index between two spectra under different illuminants.
    Returns ΔE₂₀₀₀ under D65 and A illuminant.
    """
    Lab_D65_1 = reflectance_to_Lab(R1, "D65")
    Lab_D65_2 = reflectance_to_Lab(R2, "D65")
    Lab_A_1   = reflectance_to_Lab(R1, "A")
    Lab_A_2   = reflectance_to_Lab(R2, "A")

    return {
        "dE2000_D65": delta_E_2000(Lab_D65_1, Lab_D65_2),
        "dE2000_A":   delta_E_2000(Lab_A_1,   Lab_A_2),
        "metamerism_index": abs(
            delta_E_2000(Lab_D65_1, Lab_D65_2) -
            delta_E_2000(Lab_A_1,   Lab_A_2)
        ),
    }

