"""Streaming quality checks and OD extraction for Shenzhen taxi CSV files."""
from __future__ import annotations

import csv
import datetime as dt
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


ALIASES = {
    "vehicle": ("trajectory_id", "vehicle_id", "taxi_id", "VehicleNum"),
    "time": ("date", "time", "timestamp", "Time"),
    "lon": ("lon", "lng", "longitude", "Lng"),
    "lat": ("lat", "latitude", "Lat"),
    "occupancy": ("passenger", "occupancy", "occupancy_status", "OccupancyStatus"),
    "speed": ("speed(km/h)", "speed", "Speed"),
}


def _field_map(fieldnames: Sequence[str]) -> dict[str, str]:
    available = {str(name).strip(): str(name) for name in fieldnames}
    result = {}
    for canonical, aliases in ALIASES.items():
        match = next((available[name] for name in aliases if name in available), None)
        if match is None:
            raise ValueError(f"Missing required taxi field {canonical}; observed {sorted(available)}")
        result[canonical] = match
    return result


def _timestamp(value: str) -> dt.datetime:
    text = value.strip()
    for pattern in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%H:%M:%S"):
        try:
            parsed = dt.datetime.strptime(text, pattern)
            return parsed
        except ValueError:
            pass
    raise ValueError(f"Unsupported timestamp {value!r}")


def iter_rows(paths: Iterable[Path | str]):
    for path in paths:
        with Path(path).open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fields = _field_map(reader.fieldnames or [])
            for line, row in enumerate(reader, start=2):
                try:
                    occupancy = int(float(row[fields["occupancy"]]))
                    record = {
                        "vehicle": row[fields["vehicle"]].strip(),
                        "time": _timestamp(row[fields["time"]]),
                        "lon": float(row[fields["lon"]]),
                        "lat": float(row[fields["lat"]]),
                        "occupancy": occupancy,
                        "speed_kph": float(row[fields["speed"]]),
                        "source": str(path),
                        "line": line,
                    }
                except (TypeError, ValueError) as error:
                    yield {"invalid": True, "source": str(path), "line": line, "reason": str(error)}
                    continue
                if occupancy not in (0, 1) or not all(math.isfinite(record[key]) for key in ("lon", "lat", "speed_kph")):
                    yield {"invalid": True, "source": str(path), "line": line, "reason": "non-finite value or invalid occupancy"}
                else:
                    yield record


def in_bbox(row: Mapping, bbox: Mapping[str, float]) -> bool:
    return bbox["west"] <= row["lon"] <= bbox["east"] and bbox["south"] <= row["lat"] <= bbox["north"]


def summarize(rows: Iterable[Mapping], bbox: Mapping[str, float], start_hour=10, end_hour=16) -> dict:
    total = invalid = swapped_candidate = 0
    valid, inside, period, moving = [], [], [], []
    vehicles = set()
    for row in rows:
        total += 1
        if row.get("invalid"):
            invalid += 1
            continue
        valid.append(row)
        vehicles.add(row["vehicle"])
        if bbox["west"] <= row["lat"] <= bbox["east"] and bbox["south"] <= row["lon"] <= bbox["north"]:
            swapped_candidate += 1
        if in_bbox(row, bbox):
            inside.append(row)
            hour = row["time"].hour + row["time"].minute / 60 + row["time"].second / 3600
            if start_hour <= hour < end_hour:
                period.append(row)
                if 0 < row["speed_kph"] <= 120:
                    moving.append(row["speed_kph"])
    speeds = np.asarray(moving, dtype=float)
    return {
        "rows_total": total,
        "rows_invalid": invalid,
        "rows_valid": len(valid),
        "vehicles": len(vehicles),
        "rows_in_bbox": len(inside),
        "rows_in_period": len(period),
        "moving_speed_rows": int(speeds.size),
        "coordinate_swap_candidates": swapped_candidate,
        "moving_speed_kph_p10": float(np.quantile(speeds, 0.10)) if speeds.size else None,
        "moving_speed_kph_median": float(np.median(speeds)) if speeds.size else None,
        "moving_speed_kph_p90": float(np.quantile(speeds, 0.90)) if speeds.size else None,
        "occupancy_fraction_period": float(np.mean([row["occupancy"] for row in period])) if period else None,
    }


def extract_completed_trips(rows: Iterable[Mapping], bbox: Mapping[str, float], start_hour=10, end_hour=16) -> list[dict]:
    by_vehicle = defaultdict(list)
    for row in rows:
        if not row.get("invalid") and in_bbox(row, bbox):
            by_vehicle[row["vehicle"]].append(row)
    trips = []
    for vehicle, records in by_vehicle.items():
        records.sort(key=lambda value: value["time"])
        pickup = None
        previous = None
        for current in records:
            if previous is not None and previous["occupancy"] == 0 and current["occupancy"] == 1:
                pickup = current
            elif previous is not None and previous["occupancy"] == 1 and current["occupancy"] == 0 and pickup is not None:
                hour = pickup["time"].hour + pickup["time"].minute / 60 + pickup["time"].second / 3600
                if start_hour <= hour < end_hour and current["time"] >= pickup["time"]:
                    trips.append({
                        "vehicle": vehicle,
                        "pickup_time": pickup["time"].isoformat(sep=" "),
                        "dropoff_time": current["time"].isoformat(sep=" "),
                        "pickup_lon": pickup["lon"], "pickup_lat": pickup["lat"],
                        "dropoff_lon": current["lon"], "dropoff_lat": current["lat"],
                        "duration_minutes": (current["time"] - pickup["time"]).total_seconds() / 60,
                    })
                pickup = None
            previous = current
    return trips

