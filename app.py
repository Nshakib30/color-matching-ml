"""
app.py
======
Flask web server for the Color Matching ML System.
Provides REST API endpoints for the web interface.

Run:  python app.py
Then: open http://localhost:5000

Author: ColorMatching ML Project — FZL
"""

import sys
import os
import io
import json
import numpy as np

# Force UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from src.parsing.qtx_parser import parse_standard_qtx, reflectance_columns
from src.physics.color_metrics import reflectance_to_Lab, delta_E_2000, WAVELENGTHS
from src.physics.kubelka_munk import KMDyeLibrary
from src.models.ml_models import InverseColorModel
from src.search.combo_search import ComboSearchEngine
from pathlib import Path
import pandas as pd

app = Flask(__name__, static_folder="web/static", template_folder="web")
CORS(app)

# ─── Configuration ────────────────────────────────────────────────────
DATA_DIR = Path("G:/My Drive/ColorMission Backup QTX files")
PROC_DIR = Path("C:/ColorMatchingML/data/processed")

SUBSTRATE_FILES = {
    "tape": {
        "color_book": DATA_DIR / "FZL Tape COLOR BOOK.qtx",
        "standards":  DATA_DIR / "FZL STD COLOR.qtx",
    },
    "thread": {
        "color_book1": DATA_DIR / "FZL Thread color book 1.qtx",
        "color_book2": DATA_DIR / "FZL Thread color book 2.qtx",
        "standards":   DATA_DIR / "FZL STD COLOR.qtx",
    },
}

R_COLS = [f"R{wl}" for wl in WAVELENGTHS]

# ─── Cache loaded models (load once, reuse) ───────────────────────────
_cache = {}

def get_resources(substrate: str, model_name: str = "xgboost"):
    """Load and cache KM library + inverse model + color book for a substrate."""
    key = f"{substrate}_{model_name}"
    if key not in _cache:
        print(f"  Loading resources for {substrate}/{model_name}...")

        km = KMDyeLibrary.load(str(PROC_DIR / f"km_library_{substrate}.pkl"))

        inv_path = PROC_DIR / f"inverse_{model_name}_{substrate}.pkl"
        inv = InverseColorModel.load(str(inv_path)) if inv_path.exists() else None

        # Load color book
        if substrate == "tape":
            cb = parse_standard_qtx(str(SUBSTRATE_FILES["tape"]["color_book"]))
        else:
            cb1 = parse_standard_qtx(str(SUBSTRATE_FILES["thread"]["color_book1"]))
            cb2 = parse_standard_qtx(str(SUBSTRATE_FILES["thread"]["color_book2"]))
            cb = pd.concat([cb1, cb2], ignore_index=True)

        cb["_key"] = cb["name"].str.strip().str.upper()

        search_engine = ComboSearchEngine(km, dE_threshold=1.0, use_prescreening=True)

        _cache[key] = {
            "km": km,
            "inverse": inv,
            "color_book": cb,
            "search_engine": search_engine,
            "dye_names": km.dye_names,
        }
        print(f"  Ready: {len(km.dye_names)} dyes loaded.")

    return _cache[key]


def lab_to_hex(L, a, b):
    """Convert Lab to approximate hex color for display."""
    try:
        import colour
        Lab = np.array([L, a, b])
        illum = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
        XYZ = colour.Lab_to_XYZ(Lab, illum)
        RGB = colour.XYZ_to_sRGB(XYZ)
        RGB = np.clip(RGB, 0, 1)
        r, g, b_val = (RGB * 255).astype(int)
        return f"#{r:02x}{g:02x}{b_val:02x}"
    except Exception:
        return "#cccccc"


# ─── Routes ──────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("web", "index.html")


@app.route("/api/colors", methods=["GET"])
def list_colors():
    """Return list of all available color names for a substrate."""
    substrate = request.args.get("substrate", "tape")
    try:
        res = get_resources(substrate)
        names = res["color_book"]["name"].tolist()
        return jsonify({"colors": names, "count": len(names)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/dyes", methods=["GET"])
def list_dyes():
    """Return list of available dye names for a substrate."""
    substrate = request.args.get("substrate", "tape")
    try:
        res = get_resources(substrate)
        return jsonify({"dyes": res["dye_names"], "count": len(res["dye_names"])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint.

    Body (JSON):
    {
        "substrate": "tape" | "thread",
        "model": "xgboost" | "lightgbm" | "random_forest" | ...,
        "color_name": "E001",          // OR
        "lab": [L, a, b],             // direct Lab input
        "top_n": 3
    }
    """
    data = request.get_json()
    substrate  = data.get("substrate", "tape")
    model_name = data.get("model", "xgboost")
    color_name = data.get("color_name", "").strip()
    lab_input  = data.get("lab", None)
    top_n      = int(data.get("top_n", 3))

    try:
        res = get_resources(substrate, model_name)
        km  = res["km"]
        se  = res["search_engine"]

        # ── Resolve target reflectance ──────────────────────────────
        target_R   = None
        target_Lab = None

        if color_name:
            cb = res["color_book"]
            match = cb[cb["_key"] == color_name.upper()]
            if match.empty:
                # Fuzzy: partial match
                match = cb[cb["_key"].str.contains(color_name.upper(), na=False)]
            if match.empty:
                return jsonify({"error": f"Color '{color_name}' not found in {substrate} color book."}), 404

            target_R   = match.iloc[0][R_COLS].values.astype(float)
            target_Lab = reflectance_to_Lab(target_R / 100.0)

        elif lab_input and len(lab_input) == 3:
            target_Lab = np.array(lab_input, dtype=float)
            # Find closest dye spectrum as proxy
            dye_labs = {
                d: reflectance_to_Lab(m.predict_reflectance(1.0))
                for d, m in km.models.items()
            }
            best = min(dye_labs, key=lambda d: delta_E_2000(target_Lab, dye_labs[d]))
            target_R = km.models[best].predict_reflectance(1.0) * 100.0

        else:
            return jsonify({"error": "Provide color_name or lab [L, a, b]"}), 400

        # ── Run combinatorial K-M search ───────────────────────────
        results = se.search(target_R, n_best=top_n, verbose=False)

        # ── Format output ──────────────────────────────────────────
        recipes = []
        for r in results:
            active_dyes = {
                dye: round(conc, 6)
                for dye, conc in r["recipe"].items()
                if conc > 0.0001
            }
            pred_lab = r.get("pred_Lab", [0, 0, 0])
            recipes.append({
                "rank":          len(recipes) + 1,
                "dE2000":        r["dE2000"],
                "grade":         r["grade"],
                "n_dyes":        r["n_dyes"],
                "recipe":        active_dyes,
                "predicted_lab": pred_lab,
                "predicted_hex": lab_to_hex(*pred_lab),
            })

        target_lab_list = target_Lab.tolist()
        return jsonify({
            "substrate":    substrate,
            "model":        model_name,
            "color_name":   color_name or "Custom Lab",
            "target_lab":   target_lab_list,
            "target_hex":   lab_to_hex(*target_lab_list),
            "recipes":      recipes,
            "dyes_available": res["dye_names"],
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Color Matching ML System Running"})


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  FZL Color Matching ML System")
    print("  Web Interface: http://localhost:5000")
    print("="*60 + "\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
