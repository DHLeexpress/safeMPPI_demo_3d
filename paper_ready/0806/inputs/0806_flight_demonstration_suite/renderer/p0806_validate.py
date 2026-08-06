"""Independent MP4 validation via ffprobe + hashing for the 0806 prep.

Checks every MP4 in a directory: h264 codec, yuv420p, fixed expected
resolution, constant frame rate, decodable stream. Writes validation.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def probe(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries",
         "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_frames,duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)["streams"][0]


def decode_check(path):
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
        capture_output=True, text=True,
    )
    return result.returncode == 0 and not result.stderr.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True)
    parser.add_argument("--expected-width", type=int, default=884)
    parser.add_argument("--expected-height", type=int, default=884)
    args = parser.parse_args()

    root = Path(args.dir)
    rows = []
    ok = True
    for mp4 in sorted(root.rglob("*.mp4")):
        stream = probe(mp4)
        decodable = decode_check(mp4)
        row = {
            "file": str(mp4.relative_to(root)),
            "bytes": mp4.stat().st_size,
            "sha256": hashlib.sha256(mp4.read_bytes()).hexdigest(),
            "codec": stream.get("codec_name"),
            "pix_fmt": stream.get("pix_fmt"),
            "width": stream.get("width"),
            "height": stream.get("height"),
            "avg_frame_rate": stream.get("avg_frame_rate"),
            "nb_frames": stream.get("nb_frames"),
            "duration_s": stream.get("duration"),
            "decodable": decodable,
        }
        row["valid"] = bool(
            row["codec"] == "h264"
            and row["pix_fmt"] == "yuv420p"
            and row["width"] == args.expected_width
            and row["height"] == args.expected_height
            and decodable
        )
        ok &= row["valid"]
        rows.append(row)
    report = {"all_valid": bool(ok), "count": len(rows), "videos": rows}
    (root / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"all_valid": ok, "count": len(rows)}))
    if not ok:
        raise SystemExit("MP4 validation FAILED")


if __name__ == "__main__":
    main()
