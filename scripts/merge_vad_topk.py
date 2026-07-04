"""Fuse multi-channel NeMo VAD frame scores with dynamic top-k channel selection.

NeMo VAD frame files contain one speech probability per line. This script reads
one ``vad_outputs`` directory per channel, selects the top-k channel scores at
each frame, averages them, thresholds the fused score, and writes a JSONL
manifest that can be passed to ``diarizer.vad.external_vad_manifest``.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Fuse multi-channel NeMo VAD frames with dynamic top-k.")
    parser.add_argument(
        "--vad-dirs",
        nargs="+",
        type=Path,
        required=True,
        help="Directories containing NeMo vad_outputs/*.frame files, one directory per channel.",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        required=True,
        help="Mono audio directory used for embedding extraction in the external manifest.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output external VAD JSONL manifest.")
    parser.add_argument("--pattern", default="*.frame", help="Frame filename pattern.")
    parser.add_argument("--top-k", type=int, default=2, help="Average the highest k channel scores per frame.")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Speech threshold applied to the fused top-k frame score.",
    )
    parser.add_argument("--frame-shift", type=float, default=0.01, help="Frame shift in seconds.")
    parser.add_argument("--merge-gap", type=float, default=0.2, help="Merge speech intervals separated by this gap.")
    parser.add_argument("--min-duration", type=float, default=0.05, help="Drop merged intervals shorter than this.")
    parser.add_argument("--pad-onset", type=float, default=0.2, help="Add seconds before each detected segment.")
    parser.add_argument("--pad-offset", type=float, default=0.2, help="Add seconds after each detected segment.")
    return parser.parse_args()


def read_frame_scores(path: Path):
    scores = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                scores.append(float(text))
    return scores


def collect_recordings(vad_dirs, pattern):
    by_recording = defaultdict(list)
    for vad_dir in vad_dirs:
        if not vad_dir.is_dir():
            raise FileNotFoundError(f"VAD directory not found: {vad_dir}")

        for frame_file in sorted(vad_dir.glob(pattern)):
            by_recording[frame_file.stem].append(read_frame_scores(frame_file))

    return by_recording


def topk_frame_intervals(channel_scores, top_k: int, score_threshold: float, frame_shift: float):
    if len(channel_scores) < top_k:
        raise ValueError(f"Need at least {top_k} channels, got {len(channel_scores)}")

    frame_count = min(len(scores) for scores in channel_scores)
    intervals = []
    active_start = None

    for frame_idx in range(frame_count):
        scores = sorted((scores[frame_idx] for scores in channel_scores), reverse=True)
        fused_score = sum(scores[:top_k]) / top_k
        is_speech = fused_score >= score_threshold

        if is_speech and active_start is None:
            active_start = frame_idx * frame_shift
        elif not is_speech and active_start is not None:
            intervals.append((active_start, frame_idx * frame_shift))
            active_start = None

    if active_start is not None:
        intervals.append((active_start, frame_count * frame_shift))

    return intervals


def pad_intervals(intervals, pad_onset: float, pad_offset: float):
    if pad_onset <= 0 and pad_offset <= 0:
        return intervals
    return [(max(0.0, start - pad_onset), end + pad_offset) for start, end in intervals]


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
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")
    if args.top_k > len(args.vad_dirs):
        raise ValueError("--top-k cannot be larger than the number of VAD directories")

    vad_by_recording = collect_recordings(args.vad_dirs, args.pattern)
    if not vad_by_recording:
        raise FileNotFoundError("No VAD frame files found.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_segments = 0
    total_duration = 0.0
    with args.output.open("w", encoding="utf-8") as f:
        for recording_id in sorted(vad_by_recording):
            audio_path = args.audio_dir / f"{recording_id}.flac"
            if not audio_path.is_file():
                raise FileNotFoundError(f"Audio file not found for {recording_id}: {audio_path}")

            raw = topk_frame_intervals(
                vad_by_recording[recording_id],
                top_k=args.top_k,
                score_threshold=args.score_threshold,
                frame_shift=args.frame_shift,
            )
            padded = pad_intervals(raw, args.pad_onset, args.pad_offset)
            merged = merge_intervals(padded, args.merge_gap, args.min_duration)

            for start, end in merged:
                duration = end - start
                entry = {
                    "audio_filepath": str(audio_path.resolve()).replace("\\", "/"),
                    "offset": round(start, 5),
                    "duration": round(duration, 5),
                    "label": "UNK",
                    "uniq_id": recording_id,
                }
                json.dump(entry, f, ensure_ascii=False)
                f.write("\n")
                total_duration += duration
            total_segments += len(merged)

    print(
        "Wrote "
        f"{total_segments} top-k VAD segments ({total_duration:.2f}s) for {len(vad_by_recording)} recordings "
        f"to {args.output} with top_k={args.top_k}, threshold={args.score_threshold}"
    )


if __name__ == "__main__":
    main()
