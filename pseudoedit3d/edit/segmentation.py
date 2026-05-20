from __future__ import annotations

import numpy as np


def mask_to_segments(mask: np.ndarray, min_len: int = 6) -> list[tuple[int, int]]:
    mask = mask.astype(bool)
    segments = []
    start = None
    for idx, value in enumerate(mask):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            if idx - start >= min_len:
                segments.append((start, idx - 1))
            start = None
    if start is not None and len(mask) - start >= min_len:
        segments.append((start, len(mask) - 1))
    return segments


def detect_active_span(values: np.ndarray, min_len: int = 6) -> tuple[np.ndarray, list[tuple[int, int]]]:
    centered = values - np.median(values)
    velocity = np.diff(values, prepend=values[:1])
    amp_thr = max(8.0, float(np.percentile(np.abs(centered), 70)))
    vel_thr = max(2.0, float(np.percentile(np.abs(velocity), 75)))
    active = (np.abs(centered) >= amp_thr) | (np.abs(velocity) >= vel_thr)
    if active.any():
        smoothed = np.convolve(active.astype(np.float32), np.ones((5,), dtype=np.float32), mode="same") >= 1.0
        active = smoothed.astype(bool)
    segments = mask_to_segments(active, min_len=min_len)
    if not segments:
        peak = int(np.argmax(np.abs(centered)))
        start = max(0, peak - min_len)
        end = min(len(values) - 1, peak + min_len)
        active = np.zeros((len(values),), dtype=bool)
        active[start:end + 1] = True
        segments = [(start, end)]
    return active.astype(np.float32), segments
