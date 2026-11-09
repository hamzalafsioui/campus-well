from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold,train_test_split

from . import config

def load_raw(path: str | Path | None = None):
    if path is None:
        path = config.DATA_RAW / config.RAW_FILENAME
        if not path.exists():
            csvs  =sorted(config.DATA_RAW.glob("*.csv"))
            if len(csvs) != 1:
                raise FileNotFoundError(f"{config.RAW_FILENAME} does not exists")

            path = csvs[0]
        for enc in ("utf-8","utf-8-sig","latin-1"):
            try:
                df = pd.read_csv(path,encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f"Encoding non reconnu : {path}")
        df.columns = df.columns.str.strip()

        return df



def clean_data(df:pd.DataFrame):

    out = df.copy()

    # Numeric Columns
    for col in config.NUMERIC_RANGES:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col],errors="coerce") 

    # Categorical columns
    for col,allowed in config.CATEGORICAL_VALUES.items():
        if col not in out.columns:
            continue
        mapping = {v.lower(): v for v in allowed}

        out[col] = (out[col].astype("string").str.strip().str.lower().map(mapping)
                    )

    # Target Column
    target_mapping = {v.lower(): v for v in config.TARGET_ORDER}
    out[config.TARGET_COL] = (
        out[config.TARGET_COL].astype("string").str.strip().str.lower().map(target_mapping)
    )

    return out