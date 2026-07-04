"""Fuse multi-channel NeMo VAD outputs with k-of-n voting.

NeMo VAD text files use lines in this format:

    <start_sec> <duration_sec> speech

This script reads one VAD output directory per channel, keeps time ranges where
at least ``--vote-threshold`` channels are speech, and writes a JSONL manifest
that can be passed to ``diarizer.vad.external_vad_manifest``.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Fuse multi-channel NeMo VAD outputs with k-of-n voting.")
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
    parser.add_argument(
        "--vote-threshold",
        type=int,
        default=2,
        help="Keep speech only when at least this many channels vote speech.",
    )
    parser.add_argument("--merge-gap", type=float, default=0.2, help="Merge speech intervals separated by this gap.")
    parser.add_argument("--min-duration", type=float, default=0.05, help="Drop merged intervals shorter than this.")
    parser.add_argument(
        "--pre-pad-min-duration",
        type=float,
        default=0.0,
        help="Drop voted intervals shorter than this before padding. This removes unstable short coincidences.",
    )
    parser.add_argument(
        "--pad-onset",
        type=float,
        default=0.0,
        help="Add this many seconds before each voted speech segment.",
    )
    parser.add_argument(
        "--pad-offset",
        type=float,
        default=0.0,
        help="Add this many seconds after each voted speech segment.",
    )
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


def vote_intervals(channel_intervals, vote_threshold: int):
    events = defaultdict(int)
    for intervals in channel_intervals:
        for start, end in intervals:
            if end > start:
                events[start] += 1
                events[end] -= 1

    if not events:
        return []

    voted = []
    active_votes = 0
    prev_time = None
    for time in sorted(events):
        if prev_time is not None and time > prev_time and active_votes >= vote_threshold:
            voted.append((prev_time, time))
        active_votes += events[time]
        prev_time = time
    return voted


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


def filter_intervals(intervals, min_duration: float):
    if min_duration <= 0:
        return intervals
    return [(start, end) for start, end in intervals if end - start >= min_duration]


def pad_intervals(intervals, pad_onset: float, pad_offset: float):
    if pad_onset <= 0 and pad_offset <= 0:
        return intervals
    return [(max(0.0, start - pad_onset), end + pad_offset) for start, end in intervals]


def collect_recordings(vad_dirs, pattern):
    by_recording = defaultdict(list)
    for vad_dir in vad_dirs:
        if not vad_dir.is_dir():
            raise FileNotFoundError(f"VAD directory not found: {vad_dir}")

        per_channel = {}
        for vad_file in sorted(vad_dir.glob(pattern)):
            per_channel[vad_file.stem] = read_vad_txt(vad_file)

        for recording_id, intervals in per_channel.items():
            by_recording[recording_id].append(intervals)

    return by_recording


def main() -> None:
    args = parse_args()
    if args.vote_threshold < 1:
        raise ValueError("--vote-threshold must be >= 1")
    if args.vote_threshold > len(args.vad_dirs):
        raise ValueError("--vote-threshold cannot be larger than the number of VAD directories")

    vad_by_recording = collect_recordings(args.vad_dirs, args.pattern)
    if not vad_by_recording:
        raise FileNotFoundError("No VAD txt files found.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_segments = 0
    with args.output.open("w", encoding="utf-8") as f:
        for recording_id in sorted(vad_by_recording):
            audio_path = args.audio_dir / f"{recording_id}.flac"
            if not audio_path.is_file():
                raise FileNotFoundError(f"Audio file not found for {recording_id}: {audio_path}")

            voted = vote_intervals(vad_by_recording[recording_id], args.vote_threshold)
            voted = filter_intervals(voted, args.pre_pad_min_duration)
            padded = pad_intervals(voted, args.pad_onset, args.pad_offset)
            merged = merge_intervals(padded, args.merge_gap, args.min_duration)
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

    print(
        "Wrote "
        f"{total_segments} voted VAD segments for {len(vad_by_recording)} recordings "
        f"to {args.output} with threshold {args.vote_threshold}/{len(args.vad_dirs)}"
    )


if __name__ == "__main__":
    main()
