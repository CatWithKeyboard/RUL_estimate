from __future__ import annotations

import pandas as pd


def add_rul_labels(df: pd.DataFrame, time_col: str = "timestamp") -> pd.DataFrame:
    out = df.sort_values(time_col).copy()
    failure_time = out[time_col].max()
    delta = failure_time - out[time_col]
    out["rul_seconds"] = delta.dt.total_seconds()
    out["rul_minutes"] = out["rul_seconds"] / 60.0
    out["rul_hours"] = out["rul_minutes"] / 60.0
    return out


def add_normalized_life(df: pd.DataFrame, rul_col: str = "rul_hours") -> pd.DataFrame:
    out = df.copy()
    max_rul = out[rul_col].max()
    out["life_pct_remaining"] = out[rul_col] / max_rul if max_rul > 0 else 0.0
    out["degradation_pct"] = 1.0 - out["life_pct_remaining"]
    return out
