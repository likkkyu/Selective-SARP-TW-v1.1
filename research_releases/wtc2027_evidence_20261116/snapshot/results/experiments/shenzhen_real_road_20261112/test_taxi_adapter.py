import csv
import json
import tempfile
from pathlib import Path

from taxi_adapter import extract_completed_trips, iter_rows, summarize


BBOX = {"south": 22.5, "west": 114.02, "north": 22.59, "east": 114.117}


def main():
    rows = [
        ["taxi-a", "2014-10-22 10:00:00", 114.050, 22.530, 0, 20],
        ["taxi-a", "2014-10-22 10:01:00", 114.051, 22.531, 1, 18],
        ["taxi-a", "2014-10-22 10:10:00", 114.061, 22.541, 1, 30],
        ["taxi-a", "2014-10-22 10:11:00", 114.062, 22.542, 0, 10],
        ["taxi-b", "2014-10-22 09:59:00", 114.070, 22.550, 1, 15],
        ["taxi-b", "2014-10-22 10:04:00", 114.071, 22.551, 0, 0],
    ]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "sample.csv"
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["trajectory_id", "date", "lon", "lat", "passenger", "speed(km/h)"])
            writer.writerows(rows)
        parsed = list(iter_rows([path]))
    summary = summarize(parsed, BBOX)
    trips = extract_completed_trips(parsed, BBOX)
    assert summary["rows_valid"] == 6 and summary["rows_in_period"] == 5
    assert summary["moving_speed_kph_median"] == 19.0
    assert len(trips) == 1 and trips[0]["duration_minutes"] == 10.0
    assert trips[0]["pickup_lon"] == 114.051 and trips[0]["dropoff_lon"] == 114.062
    print(json.dumps({"status": "pass", "summary": summary, "trips": trips}, indent=2))


if __name__ == "__main__":
    main()
