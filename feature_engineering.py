from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew, entropy
from scipy.signal import welch


SAMPLE_RATE = 20000


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b not in (0, 0.0) else 0.0


def extract_time_domain_features(signal: np.ndarray) -> dict:
    x = np.asarray(signal, dtype=float).ravel()
    abs_x = np.abs(x)
    rms = np.sqrt(np.mean(x ** 2))
    mean_abs = np.mean(abs_x)
    peak = np.max(abs_x)
    std = np.std(x)
    return {
        "mean": float(np.mean(x)),
        "std": float(std),
        "var": float(np.var(x)),
        "rms": float(rms),
        "peak": float(peak),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "ptp": float(np.ptp(x)),
        "skewness": float(skew(x, bias=False)),
        "kurtosis": float(kurtosis(x, fisher=False, bias=False)),
        "crest_factor": _safe_div(peak, rms),
        "shape_factor": _safe_div(rms, mean_abs),
        "impulse_factor": _safe_div(peak, mean_abs),
        "clearance_factor": _safe_div(peak, np.mean(np.sqrt(abs_x)) ** 2),
        "energy": float(np.sum(x ** 2)),
    }


def extract_frequency_domain_features(signal: np.ndarray, sample_rate: int = SAMPLE_RATE) -> dict:
    x = np.asarray(signal, dtype=float).ravel()
    freqs = np.fft.rfftfreq(len(x), d=1 / sample_rate)
    fft_mag = np.abs(np.fft.rfft(x))
    power = fft_mag ** 2
    power_sum = np.sum(power) + 1e-12
    centroid = np.sum(freqs * power) / power_sum
    dominant_idx = int(np.argmax(fft_mag))
    dominant_freq = freqs[dominant_idx]
    psd_freqs, psd = welch(x, fs=sample_rate, nperseg=min(1024, len(x)))
    psd_norm = psd / (np.sum(psd) + 1e-12)
    band_edges = [0, 1000, 3000, 5000, 10000]
    band_feats = {}
    for start, end in zip(band_edges[:-1], band_edges[1:]):
        mask = (psd_freqs >= start) & (psd_freqs < end)
        band_feats[f"band_energy_{start}_{end}"] = float(np.sum(psd[mask]))
    return {
        "spectral_centroid": float(centroid),
        "dominant_frequency": float(dominant_freq),
        "spectral_entropy": float(entropy(psd_norm + 1e-12)),
        "fft_mean": float(np.mean(fft_mag)),
        "fft_std": float(np.std(fft_mag)),
        "fft_max": float(np.max(fft_mag)),
        **band_feats,
    }


def extract_features(signal: np.ndarray, sample_rate: int = SAMPLE_RATE) -> dict:
    features = {}
    features.update(extract_time_domain_features(signal))
    features.update(extract_frequency_domain_features(signal, sample_rate=sample_rate))
    return features


def smooth_health_indicator(series: pd.Series, window: int = 5) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()
