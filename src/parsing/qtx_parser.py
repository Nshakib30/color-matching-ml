"""
qtx_parser.py
=============
Parses QTX files (INI-style spectrophotometer format) into structured DataFrames.

QTX Format:
  [STANDARD_DATA N]
  STD_NAME=DYE_NAME CONCENTRATION
  STD_REFLPOINTS=40
  STD_REFLINTERVAL=10
  STD_REFLLOW=360
  STD_R=R360, R370, ..., R750  (40 values)

Author: ColorMatching ML Project
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


# Wavelength axis: 360nm to 750nm at 10nm intervals = 40 points
WAVELENGTHS = np.arange(360, 760, 10)  # [360, 370, ..., 750]


def parse_qtx(filepath: str) -> pd.DataFrame:
    """
    Parse a QTX file and return a DataFrame with one row per measurement.

    Returns:
        DataFrame with columns:
            name        : str  — sample name (e.g., "BLACK 1.5")
            R360..R750  : float — reflectance at each wavelength (0-100 scale)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"QTX file not found: {filepath}")

    records = []
    current = {}

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()

            # New record block
            if re.match(r"^\[STANDARD_DATA\s+\d+\]", line):
                if current:
                    records.append(current)
                current = {}

            elif line.startswith("STD_NAME="):
                current["name"] = line.split("=", 1)[1].strip()

            elif line.startswith("STD_R="):
                raw = line.split("=", 1)[1].strip()
                # Values are comma-separated floats, trailing comma OK
                vals = [float(v) for v in raw.rstrip(",").split(",") if v.strip()]
                if len(vals) == 40:
                    current["reflectance"] = np.array(vals)

    # Append last record
    if current:
        records.append(current)

    # Convert to DataFrame
    rows = []
    for rec in records:
        if "name" in rec and "reflectance" in rec:
            row = {"name": rec["name"]}
            for i, wl in enumerate(WAVELENGTHS):
                row[f"R{wl}"] = rec["reflectance"][i]
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def parse_calibration_qtx(filepath: str) -> pd.DataFrame:
    """
    Parse a calibration QTX file where sample names encode:
      "<DYE_NAME> <CONCENTRATION>"
    e.g., "BLACK 1.5", "BLUE HLBT 0.025"

    Returns:
        DataFrame with columns:
            dye_name    : str  — base dye identifier
            concentration: float — concentration (% owf)
            R360..R750  : float — reflectance values
    """
    df = parse_qtx(filepath)

    dye_names = []
    concentrations = []

    for name in df["name"]:
        # Split on last whitespace token which should be the concentration
        parts = name.rsplit(" ", 1)
        if len(parts) == 2:
            try:
                conc = float(parts[1])
                dye = parts[0].strip()
            except ValueError:
                dye = name
                conc = np.nan
        else:
            dye = name
            conc = np.nan

        # Handle edge case: "RED 3BFF 30.0005" — the "3" prefix was a formatting artefact
        # The true name tokens ending in digits ≥ 30 are concentration artefacts
        # Clean them: concentration should be ≤ 10 for realistic dye % owf
        if conc is not None and conc >= 10:
            # The concentration got merged with a number in the dye name
            # e.g. "RED 3BFF 3" + "0.0005" → name="RED 3BFF 30.0005"
            # Attempt to re-split with regex
            m = re.match(r"^(.+?)\s+(\d+)(\.\d+)$", name)
            if m:
                dye = m.group(1)
                try:
                    conc = float(m.group(2)[1:] + m.group(3))  # strip leading digit
                except Exception:
                    pass

        dye_names.append(dye)
        concentrations.append(conc)

    df.insert(0, "concentration", concentrations)
    df.insert(0, "dye_name", dye_names)
    df.drop(columns=["name"], inplace=True)

    return df


def parse_standard_qtx(filepath: str) -> pd.DataFrame:
    """
    Parse a standard color QTX file (target colors, color books).
    Returns DataFrame with columns: name, R360..R750
    """
    return parse_qtx(filepath)


def reflectance_columns() -> list:
    """Return list of reflectance column names: ['R360', 'R370', ..., 'R750']"""
    return [f"R{wl}" for wl in WAVELENGTHS]


if __name__ == "__main__":
    # Quick test
    import sys
    if len(sys.argv) > 1:
        df = parse_calibration_qtx(sys.argv[1])
        print(df[["dye_name", "concentration"]].drop_duplicates().to_string())
        print(f"\nTotal records: {len(df)}")

