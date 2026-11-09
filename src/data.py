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


def split_data(df:pd.DataFrame,save:bool = True):

    data = df.copy()

    n_before = len(data)
    data = data.dropna(subset=[config.TARGET_COL])
    n_droped_target = n_before - len(data)

    n_before_dup = len(data)
    data = data.drop_duplicates()
    n_dropped_dup = n_before_dup - len(data)

    train,test = train_test_split(data,test_size=config.TEST_SIZE,stratify=data[config.TARGET_COL],random_state=config.RANDOM_STATE)

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    info = {
        "n_dropped_missing_target":int(n_droped_target),
        "n_dropped_duplicates":int(n_dropped_dup),
        "n_train":int(len(train)),
        "n_test":int(len(test)),
        "train_missing_values":(train.isna().sum().to_dict()),
        "test_missing_values":(test.isna().sum().to_dict()),
        "train_distribution":train[config.TARGET_COL].value_counts(normalize=True).round(3).to_dict(),
        "test_distribution":test[config.TARGET_COL].value_counts(normalize=True).round(3).to_dict(),

    }

    if save:
        train.to_csv(config.DATA_INTERIM/"train.csv",index=False)
        test.to_csv(config.DATA_INTERIM/"test.csv",index=False)
        (config.DATA_INTERIM / "split_info.json").write_text(json.dumps(info,indent=2),encoding="utf-8")
    return train,test,info