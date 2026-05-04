from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import pandas as pd
import numpy as np


@dataclass
class FileRecord:
    dataset: str
    file_path: Path
    timestamp: pd.Timestamp
    channel_count: int


IMS_CONFIG = {
    "set1": {
        "channels": 8,
        "failed_bearings": [3, 4],
        "bearing_channels": {1: [0, 1], 2: [2, 3], 3: [4, 5], 4: [6, 7]},
    },
    "set2": {
        "channels": 4,
        "failed_bearings": [1],
        "bearing_channels": {1: [0], 2: [1], 3: [2], 4: [3]},
    },
    "set3": {
        "channels": 4,
        "failed_bearings": [3],
        "bearing_channels": {1: [0], 2: [1], 3: [2], 4: [3]},
    },
}


def parse_ims_timestamp(file_name: str) -> pd.Timestamp:
    stem = Path(file_name).stem
    for fmt in ["%Y.%m.%d.%H.%M.%S", "%m.%d.%Y.%H.%M.%S", "%Y-%m-%d_%H-%M-%S"]:
        try:
            return pd.to_datetime(stem, format=fmt)
        except ValueError:
            continue
    return pd.to_datetime(stem, errors="coerce")


class IMSDataLoader:
    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)

    def discover_dataset_files(self) -> List[FileRecord]:
        records: List[FileRecord] = []
        for dataset_name, cfg in IMS_CONFIG.items():
            dataset_dir = self.root_dir / dataset_name
            if not dataset_dir.exists():
                continue
            for file_path in sorted(dataset_dir.iterdir()):
                if not file_path.is_file():
                    continue
                if file_path.suffix.lower() not in {"", ".txt", ".csv", ".dat"}:
                    continue
                timestamp = parse_ims_timestamp(file_path.name)
                records.append(
                    FileRecord(
                        dataset=dataset_name,
                        file_path=file_path,
                        timestamp=timestamp,
                        channel_count=cfg["channels"],
                    )
                )
        records.sort(key=lambda x: (x.dataset, x.timestamp))
        return records

    def load_signal_file(self, file_path: str | Path, expected_channels: Optional[int] = None) -> pd.DataFrame:
        file_path = Path(file_path)
        df = pd.read_csv(file_path, sep=r"\s+", header=None, engine="python")
        if expected_channels is not None and df.shape[1] != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} channels in {file_path.name}, found {df.shape[1]}"
            )
        return df

    def get_bearing_signal(self, df: pd.DataFrame, dataset: str, bearing_id: int, combine_channels: bool = True) -> np.ndarray:
        channel_idxs = IMS_CONFIG[dataset]["bearing_channels"][bearing_id]
        values = df.iloc[:, channel_idxs].to_numpy(dtype=float)
        if values.ndim == 1:
            return values
        if values.shape[1] == 1:
            return values[:, 0]
        if combine_channels:
            return np.mean(values, axis=1)
        return values.reshape(-1)

    def iter_bearing_series(self, dataset: str, bearing_id: int, combine_channels: bool = True) -> Iterable[Dict]:
        cfg = IMS_CONFIG[dataset]
        dataset_dir = self.root_dir / dataset
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        for file_path in sorted(dataset_dir.iterdir()):
            if not file_path.is_file():
                continue
            timestamp = parse_ims_timestamp(file_path.name)
            df = self.load_signal_file(file_path, expected_channels=cfg["channels"])
            signal = self.get_bearing_signal(df, dataset, bearing_id, combine_channels=combine_channels)
            yield {
                "dataset": dataset,
                "bearing_id": bearing_id,
                "timestamp": timestamp,
                "file_path": str(file_path),
                "signal": signal,
            }
