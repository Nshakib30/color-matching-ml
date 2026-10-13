# 🎨 Color Matching ML System
### Polyester Tape & Thread — Disperse Dye Recipe Predictor

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train All Models
```bash
# Train on tape data
python pipeline.py --substrate tape --train

# Train on thread data
python pipeline.py --substrate thread --train
```

### 3. Benchmark Models
```bash
python pipeline.py --substrate tape --benchmark
```

### 4. Predict a Recipe

**By color name (from color book):**
```bash
python pipeline.py --substrate tape --predict --color "E001"
python pipeline.py --substrate thread --predict --color "104"
```

**By L*a*b* values:**
```bash
python pipeline.py --substrate tape --predict --lab 45.2 12.1 -8.3
```

**Using a specific ML model:**
```bash
python pipeline.py --substrate tape --predict --color E001 --model xgboost
python pipeline.py --substrate tape --predict --color E001 --model lightgbm
python pipeline.py --substrate tape --predict --color E001 --model random_forest
```

**Return top 5 recipes:**
```bash
python pipeline.py --substrate tape --predict --color E001 --top 5
```

---

## Project Structure

```
C:\ColorMatchingML\
│
├── pipeline.py                   ← Main entry point (CLI)
├── requirements.txt
│
├── src/
│   ├── parsing/
│   │   └── qtx_parser.py         ← QTX file parser
│   ├── physics/
│   │   ├── kubelka_munk.py       ← K-M physics layer
│   │   └── color_metrics.py      ← Lab, ΔE2000, K/S functions
│   ├── models/
│   │   └── ml_models.py          ← RF, XGBoost, LightGBM, MLP
│   └── search/
│       └── combo_search.py       ← Combinatorial 3-dye search
│
├── data/
│   └── processed/                ← Saved models (.pkl files)
│
└── outputs/                      ← Benchmark CSVs, prediction logs
```

---

## Data Files Used

| File | Purpose |
|---|---|
| `TAPE DYES Calibration.qtx` | K/S calibration curves for tape dyes |
| `THREAD DYES Calibration.qtx` | K/S calibration curves for thread dyes |
| `Tape COLOR BOOK.qtx` | Measured spectra of tape color standards |
| `Thread color book 1+2.qtx` | Measured spectra of thread color standards |
| `tape_recipes_final.csv` | 600 tape recipes with dye concentrations |
| `thread_recipes_final.csv` | 1112 thread recipes with dye concentrations |
| `STD COLOR.qtx` | Reference standard colors |

---

## Base Dyes (18 dyes identified in calibration)

| # | Dye Name | Available in |
|---|---|---|
| 1 | BLACK | Tape & Thread |
| 2 | BLUE HLBT | Tape & Thread |
| 3 | BLUE SE 2BT | Tape & Thread |
| 4 | NAVY BLUE | Tape & Thread |
| 5 | ORANGE | Tape & Thread |
| 6 | RED 3BFF | Tape & Thread |
| 7 | RED BWFT | Tape & Thread |
| 8 | RED E FBNT | Tape & Thread |
| 9 | RED GFF | Tape & Thread |
| 10 | RED R | Tape & Thread |
| 11 | RED -SD | Tape & Thread |
| 12 | RED SD NEW | Tape & Thread |
| 13 | TURQUOISE BGF | Tape & Thread |
| 14 | VIOLET | Tape & Thread |
| 15 | YELLOW 10G | Tape & Thread |
| 16 | YELLOW SPRTS | Tape & Thread |
| 17 | YELLOW-4G | Tape & Thread |

---

## Algorithm Overview

```
Target Color (Name / Lab / QTX)
         │
         ▼
    Parse Reflectance R(λ) [360-750nm, 40 points]
         │
         ▼
    Convert to CIE L*a*b* (D65 illuminant, 2° observer)
         │
    ┌────┴────────────────────────────┐
    │                                 │
    ▼                                 ▼
[K-M Physics Search]          [ML Inverse Model]
C(18,3) = 816 combos          Random Forest /
hue prescreening →             XGBoost /
~56 combos checked             LightGBM /
scipy.optimize per combo       MLP Neural Net
    │                                 │
    └────────────┬────────────────────┘
                 ▼
         Best Recipe (ΔE₂₀₀₀ ranked)
         ─────────────────────────────
         Priority 1: 3-dye recipe
         Fallback 2: 2-dye recipe
         Fallback 3: 4-dye recipe
                 │
                 ▼
         Output: Dye Concentrations (% owf)
```

---

## ΔE₂₀₀₀ Grades

| ΔE₂₀₀₀ | Grade | Meaning |
|---|---|---|
| < 1.0 | ✅ Excellent | Imperceptible difference |
| 1.0–2.0 | 🟢 Good | Perceptible only by expert |
| 2.0–3.5 | 🟡 Acceptable | Noticeable on close inspection |
| > 3.5 | 🔴 Reject | Clearly different |

---

*ColorMission — ML Color Matching System*
