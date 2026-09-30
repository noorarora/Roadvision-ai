from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GPSPoint:
    lat: float
    lon: float


class GPSTrack:
    """Timestamp-aligned GPS track using linear interpolation."""

    def __init__(self, csv_path: str | Path):
        df = pd.read_csv(csv_path)
        required = {"time_s", "lat", "lon"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"GPS CSV missing columns: {sorted(missing)}")
        if len(df) == 0:
            raise ValueError("GPS CSV is empty")
        self.time_s = df["time_s"].astype(float).to_numpy()
        self.lat = df["lat"].astype(float).to_numpy()
        self.lon = df["lon"].astype(float).to_numpy()
        order = np.argsort(self.time_s)
        self.time_s = self.time_s[order]
        self.lat = self.lat[order]
        self.lon = self.lon[order]

    def at(self, timestamp_s: float) -> Optional[GPSPoint]:
        if len(self.time_s) == 0:
            return None
        t = float(np.clip(timestamp_s, self.time_s[0], self.time_s[-1]))
        return GPSPoint(
            lat=float(np.interp(t, self.time_s, self.lat)),
            lon=float(np.interp(t, self.time_s, self.lon)),
        )
