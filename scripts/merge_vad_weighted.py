"""Fuse multi-channel NeMo VAD frame scores with reliability-aware weights.

NeMo VAD frame files contain one speech probability per line. This script reads
one ``vad_outputs`` directory per channel, estimates a channel reliability score
from each channel's frame-score distribution, applies a weighted average at each
frame, thresholds the fused score, and writes a JSONL manifest that can be
passed to ``diarizer.vad.external_vad_manifest``.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Fuse multi-channel NeMo VAD frames with reliability weights.")
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
    parser.add_argument("--weights-output", type=Path, default=None, help="Optional JSON file for learned weights.")
    parser.add_argument("--pattern", default="*.frame", help="Frame filename pattern.")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.3,
        help="Speech threshold applied to the weighted fused frame score.",
    )
    parser.add_argument(
        "--vad-speech-threshold",
        type=float,
        default=0.5,
        help="Frame score threshold used only for estimating each channel's speech ratio.",
    )
    parser.add_argument("--frame-shift", type=float, default=0.01, help="Frame shift in seconds.")
    parser.add_argument("--merge-gap", type=float, default=0.2, help="Merge speech intervals separated by this gap.")
    parser.add_argument("--min-duration", type=float, default=0.05, help="Drop merged intervals shorter than this.")
    parser.add_argument("--pad-onset", type=float, default=0.2, help="Add seconds before each detected segment.")
    parser.add_argument("--pad-offset", type=float, default=0.2, help="Add seconds after each detected segment.")
    parser.add_argument(
        "--high-fraction",
        type=float,
        default=0.2,
        help="Use the mean of the highest fraction of frame scores as the speech-confidence statistic.",
    )
    parser.add_argument(
        "--low-fraction",
        type=float,
        default=0.5,
        help="Use the mean of the lowest fraction of frame scores as the noise-floor statistic.",
    )
    parser.add_argument(
        "--weight-floor",
        type=float,
        default=0.02,
        help="Small floor added to reliability scores before normalizing weights.",
    )
    parser.add_argument("--margin-weight", type=float, default=1.0, help="Weight for high-low confidence margin.")
    parser.add_argument("--speech-ratio-weight", type=float, default=0.3, help="Weight for channel speech ratio.")
    parser.add_argument("--stability-weight", type=float, default=0.3, help="Weight for VAD confidence stability.")
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
    for channel_idx, vad_dir in enumerate(vad_dirs):
        if not vad_dir.is_dir():
            raise FileNotFoundError(f"VAD directory not found: {vad_dir}")

        for frame_file in sorted(vad_dir.glob(pattern)):
            by_recording[frame_file.stem].append(
                {
                    "channel": channel_idx,
                    "path": str(frame_file),
                    "scores": read_frame_scores(frame_file),
                }
            )

    return by_recording


def mean(values):
    return sum(values) / len(values) if values else 0.0


def mean_abs_diff(values):
    if len(values) < 2:
        return 0.0
    return sum(abs(values[idx] - values[idx - 1]) for idx in range(1, len(values))) / (len(values) - 1)


def estimate_weights(
    channel_entries,
    high_fraction: float,
    low_fraction: float,
    weight_floor: float,
    vad_speech_threshold: float,
    margin_weight: float,
    speech_ratio_weight: float,
    stability_weight: float,
):
    if not 0 < high_fraction <= 1:
        raise ValueError("--high-fraction must be in (0, 1]")
    if not 0 < low_fraction <= 1:
        raise ValueError("--low-fraction must be in (0, 1]")

    stats = []
    raw_weights = []
    for entry in channel_entries:
        scores = sorted(entry["scores"])
        if not scores:
            raise ValueError(f"Empty frame file: {entry['path']}")

        high_count = max(1, int(len(scores) * high_fraction))
        low_count = max(1, int(len(scores) * low_fraction))
        low_mean = mean(scores[:low_count])
        high_mean = mean(scores[-high_count:])
        confidence_margin = max(0.0, high_mean - low_mean)
        speech_ratio = sum(score >= vad_speech_threshold for score in entry["scores"]) / len(entry["scores"])
        jitter = mean_abs_diff(entry["scores"])
        stability = max(0.0, 1.0 - jitter)
        quality = (
            margin_weight * confidence_margin
            + speech_ratio_weight * speech_ratio
            + stability_weight * stability
        )
        raw_weight = max(0.0, quality) + weight_floor
        raw_weights.append(raw_weight)
        stats.append(
            {
                "channel": entry["channel"],
                "path": entry["path"],
                "low_mean": round(low_mean, 6),
                "high_mean": round(high_mean, 6),
                "confidence_margin": round(confidence_margin, 6),
                "speech_ratio": round(speech_ratio, 6),
                "stability": round(stability, 6),
                "quality": round(quality, 6),
            }
        )

    total = sum(raw_weights)
    if total <= 0:
        weights = [1.0 / len(raw_weights)] * len(raw_weights)
    else:
        weights = [weight / total for weight in raw_weights]

    for stat, weight in zip(stats, weights):
        stat["weight"] = round(weight, 6)

    return weights, stats


def weighted_frame_intervals(channel_entries, weights, score_threshold: float, frame_shift: float):
    frame_count = min(len(entry["scores"]) for entry in channel_entries)
    intervals = []
    active_start = None

    for frame_idx in range(frame_count):
        fused_score = 0.0
        for entry, weight in zip(channel_entries, weights):
            fused_score += weight * entry["scores"][frame_idx]
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
    vad_by_recording = collect_recordings(args.vad_dirs, args.pattern)
    if not vad_by_recording:
        raise FileNotFoundError("No VAD frame files found.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    weight_report = {}
    total_segments = 0
    total_duration = 0.0

    with args.output.open("w", encoding="utf-8") as f:
        for recording_id in sorted(vad_by_recording):
            audio_path = args.audio_dir / f"{recording_id}.flac"
            if not audio_path.is_file():
                raise FileNotFoundError(f"Audio file not found for {recording_id}: {audio_path}")

            channel_entries = sorted(vad_by_recording[recording_id], key=lambda item: item["channel"])
            weights, stats = estimate_weights(
                channel_entries,
                high_fraction=args.high_fraction,
                low_fraction=args.low_fraction,
                weight_floor=args.weight_floor,
                vad_speech_threshold=args.vad_speech_threshold,
                margin_weight=args.margin_weight,
                speech_ratio_weight=args.speech_ratio_weight,
                stability_weight=args.stability_weight,
            )
            weight_report[recording_id] = stats

            raw = weighted_frame_intervals(
                channel_entries,
                weights=weights,
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

    if args.weights_output is not None:
        args.weights_output.parent.mkdir(parents=True, exist_ok=True)
        with args.weights_output.open("w", encoding="utf-8") as f:
            json.dump(weight_report, f, ensure_ascii=False, indent=2)

    print(
        "Wrote "
        f"{total_segments} weighted VAD segments ({total_duration:.2f}s) for {len(vad_by_recording)} recordings "
        f"to {args.output} with threshold={args.score_threshold}"
    )
    for recording_id, stats in weight_report.items():
        weights_text = ", ".join(f"ch{item['channel']}={item['weight']:.3f}" for item in stats)
        print(f"{recording_id} weights: {weights_text}")


if __name__ == "__main__":
    main()
