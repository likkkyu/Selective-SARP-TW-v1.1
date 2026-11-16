"""Download the frozen current-OSM P1 bounding box from Overpass."""
import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path


DEFAULT_BBOX = (22.5, 114.02, 22.59, 114.117)
ENDPOINT = "https://overpass-api.de/api/interpreter"


def query(bbox):
    south, west, north, east = bbox
    return f"""[out:json][timeout:180];
(
  way[\"highway\"]({south},{west},{north},{east});
);
out body;
>;
out skel qt;
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default=ENDPOINT)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an existing raw response")
    request = urllib.request.Request(
        args.endpoint,
        data=urllib.parse.urlencode({"data": query(DEFAULT_BBOX)}).encode(),
        headers={"User-Agent": "Selective-SARP-TW academic preflight/1.0"},
    )
    with urllib.request.urlopen(request, timeout=240) as response:
        payload = json.load(response)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(payload, handle, separators=(",", ":"))
    print(json.dumps({"output": str(args.output), "elements": len(payload.get("elements", []))}))


if __name__ == "__main__":
    main()

