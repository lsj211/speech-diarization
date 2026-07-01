"""Merge per-channel NeMo VAD outputs into an external VAD manifest.

NeMo VAD text files use lines in this format:

    <start_sec> <duration_sec> speech

This script reads the same recording's VAD files from multiple directories,
takes the union of all detected speech intervals, merges overlapping or nearby
intervals, and writes a JSONL manifest that can be passed to
``diarizer.vad.external_vad_manifest``.
"""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Union multi-channel NeMo VAD outputs.")
    parser.add_argument(
        "--vad-dirs",
        nargs="+",
        type=Path,
        required=True,
        help="Directories containing NeMo vad_outputs/*.txt files, one directory per channel.",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        required=True,
        help="Mono audio directory used for embedding extraction in the external manifest.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output external VAD JSONL manifest.")
    parser.add_argument("--pattern", default="*.txt", help="VAD txt filename pattern.")
    parser.add_argument("--merge-gap", type=float, default=0.2, help="Merge speech intervals separated by this gap.")
    parser.add_argument("--min-duration", type=float, default=0.05, help="Drop merged intervals shorter than this.")
    return parser.parse_args()


def read_vad_txt(path: Path):
    intervals = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            start = float(parts[0])
            duration = float(parts[1])
            if duration > 0:
                intervals.append((start, start + duration))
    return intervals


def merge_intervals(intervals, merge_gap: float, min_duration: float):
    if not intervals:
        return []

    intervals = sorted(intervals)
    merged = []
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= cur_end + merge_gap:
            cur_end = max(cur_end, end)
        else:
            if cur_end - cur_start >= min_duration:
                merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end

    if cur_end - cur_start >= min_duration:
        merged.append((cur_start, cur_end))
    return merged


def main() -> None:
    args = parse_args()
    vad_by_recording = {}

    for vad_dir in args.vad_dirs:
        if not vad_dir.is_dir():
            raise FileNotFoundError(f"VAD directory not found: {vad_dir}")
        for vad_file in sorted(vad_dir.glob(args.pattern)):
            recording_id = vad_file.stem
            vad_by_recording.setdefault(recording_id, []).extend(read_vad_txt(vad_file))

    if not vad_by_recording:
        raise FileNotFoundError("No VAD txt files found.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        total_segments = 0
        for recording_id in sorted(vad_by_recording):
            audio_path = args.audio_dir / f"{recording_id}.flac"
            if not audio_path.is_file():
                raise FileNotFoundError(f"Audio file not found for {recording_id}: {audio_path}")

            merged = merge_intervals(vad_by_recording[recording_id], args.merge_gap, args.min_duration)
            for start, end in merged:
                entry = {
                    "audio_filepath": str(audio_path.resolve()).replace("\\", "/"),
                    "offset": round(start, 5),
                    "duration": round(end - start, 5),
                    "label": "UNK",
                    "uniq_id": recording_id,
                }
                json.dump(entry, f, ensure_ascii=False)
                f.write("\n")
            total_segments += len(merged)

    print(f"Wrote {total_segments} union VAD segments for {len(vad_by_recording)} recordings to {args.output}")


if __name__ == "__main__":
    main()
