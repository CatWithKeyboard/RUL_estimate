from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import pandas as pd

from data_loader import IMSDataLoader, parse_ims_timestamp
from feature_engineering import extract_features


FEATURE_META = {"dataset", "bearing_id", "timestamp", "file_path"}


def predict_folder(model_path: str | Path, dataset: str, bearing_id: int, folder_path: str | Path) -> pd.DataFrame:
    loader = IMSDataLoader(folder_path)
    model = joblib.load(model_path)
    rows = []
    target_folder = Path(folder_path)
    channel_count = 8 if dataset == "set1" else 4
    for file_path in sorted(target_folder.iterdir()):
        if not file_path.is_file():
            continue
        df = loader.load_signal_file(file_path, expected_channels=channel_count)
        signal = loader.get_bearing_signal(df, dataset, bearing_id, combine_channels=True)
        feats = extract_features(signal)
        row = {
            "dataset": dataset,
            "bearing_id": bearing_id,
            "timestamp": parse_ims_timestamp(file_path.name),
            "file_path": str(file_path),
            **feats,
        }
        rows.append(row)
    pred_df = pd.DataFrame(rows).sort_values("timestamp")
    feature_cols = [c for c in pred_df.columns if c not in FEATURE_META]
    pred_df["predicted_rul_hours"] = model.predict(pred_df[feature_cols])
    pred_df["predicted_rul_days"] = pred_df["predicted_rul_hours"] / 24.0
    return pred_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset", required=True, choices=["set1", "set2", "set3"])
    parser.add_argument("--bearing-id", required=True, type=int)
    parser.add_argument("--folder-path", required=True)
    parser.add_argument("--output-csv", default="predictions.csv")
    args = parser.parse_args()

    pred_df = predict_folder(args.model_path, args.dataset, args.bearing_id, args.folder_path)
    pred_df.to_csv(args.output_csv, index=False)
    print(pred_df[["timestamp", "predicted_rul_hours", "predicted_rul_days"]].tail())


if __name__ == "__main__":
    main()
