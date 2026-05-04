from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor  # ← GPU‑enabled regressor

from data_loader import IMSDataLoader, IMS_CONFIG
from feature_engineering import extract_features
from labeling import add_rul_labels, add_normalized_life


FEATURE_EXCLUDE = {
    "dataset", "bearing_id", "timestamp", "file_path", "rul_seconds", "rul_minutes", "rul_hours",
    "life_pct_remaining", "degradation_pct"
}


def build_feature_table(data_root: str | Path, combine_channels: bool = True) -> pd.DataFrame:
    loader = IMSDataLoader(data_root)
    rows = []
    for dataset, cfg in IMS_CONFIG.items():
        for bearing_id in cfg["failed_bearings"]:
            for item in loader.iter_bearing_series(dataset, bearing_id, combine_channels=combine_channels):
                feats = extract_features(item["signal"])
                rows.append({
                    "dataset": dataset,
                    "bearing_id": bearing_id,
                    "timestamp": item["timestamp"],
                    "file_path": item["file_path"],
                    **feats,
                })
    feature_df = pd.DataFrame(rows)
    if feature_df.empty:
        raise ValueError("No training data found. Check dataset structure: root/set1, root/set2, root/set3")
    labeled = feature_df.groupby(["dataset", "bearing_id"], group_keys=False).apply(add_rul_labels)
    labeled = add_normalized_life(labeled)
    return labeled.reset_index(drop=True)


def train_model(feature_df: pd.DataFrame, target_col: str = "rul_hours"):
    feature_cols = [c for c in feature_df.columns if c not in FEATURE_EXCLUDE]
    X = feature_df[feature_cols]
    y = feature_df[target_col]
    groups = feature_df["dataset"].astype(str) + "_b" + feature_df["bearing_id"].astype(str)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), feature_cols)
        ]
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        # XGBoost with GPU:
        ("regressor", XGBRegressor(
            tree_method="gpu_hist",   # ← GPU training
            gpu_id=0,                 # GPU index
            booster="gbtree",
            n_estimators=300,
            max_depth=12,
            learning_rate=0.1,
            random_state=42,
        )),
    ])

    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    preds = model.predict(X.iloc[test_idx])
    metrics = {
        "mae_hours": float(mean_absolute_error(y.iloc[test_idx], preds)),
        "rmse_hours": float(mean_squared_error(y.iloc[test_idx], preds) ** 0.5),
        "r2": float(r2_score(y.iloc[test_idx], preds)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "features": feature_cols,
    }
    return model, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="Root folder containing set1, set2, set3")
    parser.add_argument("--output-dir", default="artifacts", help="Where to save model and features")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_df = build_feature_table(args.data_root)
    feature_df.to_csv(output_dir / "features.csv", index=False)

    model, metrics = train_model(feature_df)
    joblib.dump(model, output_dir / "bearing_rul_model.joblib")
    pd.DataFrame([metrics]).to_json(output_dir / "metrics.json", orient="records", indent=2)
    print(metrics)


if __name__ == "__main__":
    main()
