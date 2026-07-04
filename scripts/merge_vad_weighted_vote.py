"""Fuse multi-channel NeMo VAD segments with reliability-weighted voting.

This script estimates channel reliability from per-channel ``*.frame`` scores,
then applies those weights to post-processed NeMo VAD ``*.txt`` segments. It is
designed as a reliability-aware version of k-of-n voting: when the sum of active
channel weights exceeds ``--active-weight-threshold``, the fused output is
speech.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Fuse multi-channel VAD with reliability-weighted voting.")
    parser.add_argument(
        "--vad-dirs",
        nargs="+",
        type=Path,
        required=True,
        help="Directories containing NeMo vad_outputs/*.txt files, one directory per channel.",
    )
    parser.add_argument(
        "--frame-dirs",
        nargs="+",
        type=Path,
        required=True,
        help="Directories containing NeMo vad_outputs/*.frame files, aligned with --vad-dirs.",
    )
    parser.add_argument("--audio-dir", type=Path, required=True, help="Mono audio directory for output manifest.")
    parser.add_argument(
        "--multichannel-audio-dir",
        type=Path,
        default=None,
        help="Optional original multi-channel audio directory for RMS energy and SNR reliability stats.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output external VAD JSONL manifest.")
    parser.add_argument("--weights-output", type=Path, default=None, help="Optional JSON file for channel weights.")
    parser.add_argument("--vad-pattern", default="*.txt", help="VAD txt filename pattern.")
    parser.add_argument("--frame-pattern", default="*.frame", help="Frame filename pattern.")
    parser.add_argument(
        "--active-weight-threshold",
        type=float,
        default=0.25,
        help="Keep speech when active channel weights sum to at least this value.",
    )
    parser.add_argument("--merge-gap", type=float, default=0.2, help="Merge speech intervals separated by this gap.")
    parser.add_argument("--min-duration", type=float, default=0.05, help="Drop merged intervals shorter than this.")
    parser.add_argument("--pad-onset", type=float, default=0.2, help="Add seconds before each detected segment.")
    parser.add_argument("--pad-offset", type=float, default=0.2, help="Add seconds after each detected segment.")
    parser.add_argument("--vad-speech-threshold", type=float, default=0.5, help="Threshold for speech-ratio stats.")
    parser.add_argument("--high-fraction", type=float, default=0.2, help="Top fraction for high-confidence mean.")
    parser.add_argument("--low-fraction", type=float, default=0.5, help="Bottom fraction for noise-floor mean.")
    parser.add_argument("--weight-floor", type=float, default=0.02, help="Floor before normalizing weights.")
    parser.add_argument("--margin-weight", type=float, default=1.0, help="Weight for confidence margin.")
    parser.add_argument("--speech-ratio-weight", type=float, default=0.3, help="Weight for speech ratio.")
    parser.add_argument("--stability-weight", type=float, default=0.3, help="Weight for confidence stability.")
    parser.add_argument("--energy-weight", type=float, default=0.3, help="Weight for RMS energy reliability.")
    parser.add_argument("--snr-weight", type=float, default=0.5, help="Weight for speech/non-speech SNR reliability.")
    parser.add_argument("--audio-block-size", type=int, default=65536, help="Samples per block for audio stats.")
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


def read_frame_scores(path: Path):
    scores = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                scores.append(float(text))
    return scores


def mean(values):
    return sum(values) / len(values) if values else 0.0


def mean_abs_diff(values):
    if len(values) < 2:
        return 0.0
    return sum(abs(values[idx] - values[idx - 1]) for idx in range(1, len(values))) / (len(values) - 1)


def min_max_norm(values_by_channel, default=0.0):
    finite_values = [value for value in values_by_channel.values() if value is not None and math.isfinite(value)]
    if not finite_values:
        return {channel: default for channel in values_by_channel}

    min_value = min(finite_values)
    max_value = max(finite_values)
    if max_value - min_value < 1e-12:
        return {channel: 0.5 for channel in values_by_channel}

    return {
        channel: (value - min_value) / (max_value - min_value) if value is not None else default
        for channel, value in values_by_channel.items()
    }


def estimate_audio_quality(audio_path: Path, channel_data, block_size: int):
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "Audio quality stats require numpy and soundfile. "
            "Omit --multichannel-audio-dir to use VAD-only reliability."
        ) from exc

    channel_indices = sorted(channel_data)
    totals = {
        channel: {"total_sq": 0.0, "total_count": 0, "speech_sq": 0.0, "speech_count": 0}
        for channel in channel_indices
    }

    with sf.SoundFile(audio_path) as snd:
        sample_rate = snd.samplerate
        audio_channels = snd.channels
        intervals_by_channel = {}
        for channel in channel_indices:
            sample_intervals = []
            for start, end in channel_data[channel]["intervals"]:
                start_sample = max(0, int(round(start * sample_rate)))
                end_sample = max(start_sample, int(round(end * sample_rate)))
                if end_sample > start_sample:
                    sample_intervals.append((start_sample, end_sample))
            intervals_by_channel[channel] = sample_intervals

        positions = {channel: 0 for channel in channel_indices}
        block_start = 0
        while True:
            block = snd.read(block_size, dtype="float32", always_2d=True)
            if len(block) == 0:
                break

            block_len = len(block)
            block_end = block_start + block_len
            squared = block * block

            for channel in channel_indices:
                if channel >= audio_channels:
                    continue

                channel_sq = squared[:, channel]
                total_sq = float(channel_sq.sum(dtype=np.float64))
                totals[channel]["total_sq"] += total_sq
                totals[channel]["total_count"] += block_len

                intervals = intervals_by_channel[channel]
                pos = positions[channel]
                while pos < len(intervals) and intervals[pos][1] <= block_start:
                    pos += 1
                positions[channel] = pos

                mask = np.zeros(block_len, dtype=bool)
                cur = pos
                while cur < len(intervals) and intervals[cur][0] < block_end:
                    start_sample, end_sample = intervals[cur]
                    local_start = max(start_sample, block_start) - block_start
                    local_end = min(end_sample, block_end) - block_start
                    if local_end > local_start:
                        mask[local_start:local_end] = True
                    cur += 1

                speech_count = int(mask.sum())
                if speech_count:
                    totals[channel]["speech_sq"] += float(channel_sq[mask].sum(dtype=np.float64))
                    totals[channel]["speech_count"] += speech_count

            block_start = block_end

    audio_stats = {}
    for channel in channel_indices:
        total_count = totals[channel]["total_count"]
        speech_count = totals[channel]["speech_count"]
        noise_count = max(0, total_count - speech_count)
        total_sq = totals[channel]["total_sq"]
        speech_sq = totals[channel]["speech_sq"]
        noise_sq = max(0.0, total_sq - speech_sq)

        total_power = total_sq / total_count if total_count else 0.0
        speech_power = speech_sq / speech_count if speech_count else 0.0
        noise_power = noise_sq / noise_count if noise_count else 0.0
        snr_db = 10.0 * math.log10((speech_power + 1e-12) / (noise_power + 1e-12))
        audio_stats[channel] = {
            "rms": math.sqrt(total_power),
            "speech_rms": math.sqrt(speech_power),
            "noise_rms": math.sqrt(noise_power),
            "snr_db": snr_db,
            "speech_samples": speech_count,
            "noise_samples": noise_count,
        }

    return audio_stats


def collect_by_recording(vad_dirs, frame_dirs, vad_pattern, frame_pattern):
    if len(vad_dirs) != len(frame_dirs):
        raise ValueError("--vad-dirs and --frame-dirs must have the same length")

    by_recording = defaultdict(dict)
    for channel_idx, (vad_dir, frame_dir) in enumerate(zip(vad_dirs, frame_dirs)):
        if not vad_dir.is_dir():
            raise FileNotFoundError(f"VAD directory not found: {vad_dir}")
        if not frame_dir.is_dir():
            raise FileNotFoundError(f"Frame directory not found: {frame_dir}")

        for vad_file in sorted(vad_dir.glob(vad_pattern)):
            by_recording[vad_file.stem].setdefault(channel_idx, {})["intervals"] = read_vad_txt(vad_file)
        for frame_file in sorted(frame_dir.glob(frame_pattern)):
            by_recording[frame_file.stem].setdefault(channel_idx, {})["scores"] = read_frame_scores(frame_file)

    return by_recording


def estimate_weights(channel_data, args, audio_stats=None):
    stats = []
    raw_weights = []
    for channel_idx in sorted(channel_data):
        raw_scores = channel_data[channel_idx]["scores"]
        if not raw_scores:
            raise ValueError(f"Empty frame scores for channel {channel_idx}")
        scores = sorted(raw_scores)
        high_count = max(1, int(len(scores) * args.high_fraction))
        low_count = max(1, int(len(scores) * args.low_fraction))
        low_mean = mean(scores[:low_count])
        high_mean = mean(scores[-high_count:])
        confidence_margin = max(0.0, high_mean - low_mean)
        speech_ratio = sum(score >= args.vad_speech_threshold for score in raw_scores) / len(scores)
        jitter = mean_abs_diff(raw_scores)
        stability = max(0.0, 1.0 - jitter)
        stats.append(
            {
                "channel": channel_idx,
                "low_mean": round(low_mean, 6),
                "high_mean": round(high_mean, 6),
                "confidence_margin": round(confidence_margin, 6),
                "speech_ratio": round(speech_ratio, 6),
                "stability": round(stability, 6),
            }
        )

    if audio_stats:
        energy_norm = min_max_norm({channel: audio_stats.get(channel, {}).get("rms") for channel in channel_data})
        snr_norm = min_max_norm({channel: audio_stats.get(channel, {}).get("snr_db") for channel in channel_data})
    else:
        energy_norm = {channel: 0.0 for channel in channel_data}
        snr_norm = {channel: 0.0 for channel in channel_data}

    for stat in stats:
        channel_idx = stat["channel"]
        stat["energy_norm"] = round(energy_norm[channel_idx], 6)
        stat["snr_norm"] = round(snr_norm[channel_idx], 6)
        if audio_stats and channel_idx in audio_stats:
            stat["rms"] = round(audio_stats[channel_idx]["rms"], 8)
            stat["speech_rms"] = round(audio_stats[channel_idx]["speech_rms"], 8)
            stat["noise_rms"] = round(audio_stats[channel_idx]["noise_rms"], 8)
            stat["snr_db"] = round(audio_stats[channel_idx]["snr_db"], 4)

        quality = (
            args.margin_weight * stat["confidence_margin"]
            + args.speech_ratio_weight * stat["speech_ratio"]
            + args.stability_weight * stat["stability"]
            + args.energy_weight * energy_norm[channel_idx]
            + args.snr_weight * snr_norm[channel_idx]
        )
        stat["quality"] = round(quality, 6)
        raw_weights.append(max(0.0, quality) + args.weight_floor)

    total = sum(raw_weights)
    weights = [weight / total for weight in raw_weights] if total > 0 else [1.0 / len(raw_weights)] * len(raw_weights)
    for stat, weight in zip(stats, weights):
        stat["weight"] = round(weight, 6)
    return {stat["channel"]: weight for stat, weight in zip(stats, weights)}, stats


def weighted_vote_intervals(channel_data, weights, active_weight_threshold: float):
    events = defaultdict(float)
    for channel_idx, data in channel_data.items():
        weight = weights[channel_idx]
        for start, end in data["intervals"]:
            if end > start:
                events[start] += weight
                events[end] -= weight

    intervals = []
    active_weight = 0.0
    prev_time = None
    for time in sorted(events):
        if prev_time is not None and time > prev_time and active_weight >= active_weight_threshold:
            intervals.append((prev_time, time))
        active_weight += events[time]
        prev_time = time
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
    by_recording = collect_by_recording(args.vad_dirs, args.frame_dirs, args.vad_pattern, args.frame_pattern)
    if not by_recording:
        raise FileNotFoundError("No VAD files found.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    weight_report = {}
    total_segments = 0
    total_duration = 0.0

    with args.output.open("w", encoding="utf-8") as f:
        for recording_id in sorted(by_recording):
            audio_path = args.audio_dir / f"{recording_id}.flac"
            if not audio_path.is_file():
                raise FileNotFoundError(f"Audio file not found for {recording_id}: {audio_path}")

            channel_data = by_recording[recording_id]
            missing = [idx for idx, data in channel_data.items() if "intervals" not in data or "scores" not in data]
            if missing:
                raise ValueError(f"Missing VAD txt or frame scores for {recording_id}, channels: {missing}")

            audio_stats = None
            if args.multichannel_audio_dir is not None:
                source_audio_path = args.multichannel_audio_dir / f"{recording_id}.flac"
                if not source_audio_path.is_file():
                    raise FileNotFoundError(
                        f"Multi-channel audio file not found for {recording_id}: {source_audio_path}"
                    )
                audio_stats = estimate_audio_quality(source_audio_path, channel_data, args.audio_block_size)

            weights, stats = estimate_weights(channel_data, args, audio_stats)
            weight_report[recording_id] = stats

            raw = weighted_vote_intervals(channel_data, weights, args.active_weight_threshold)
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
        f"{total_segments} weighted-vote VAD segments ({total_duration:.2f}s) for {len(by_recording)} recordings "
        f"to {args.output} with active_weight_threshold={args.active_weight_threshold}"
    )
    for recording_id, stats in weight_report.items():
        weights_text = ", ".join(f"ch{item['channel']}={item['weight']:.3f}" for item in stats)
        print(f"{recording_id} weights: {weights_text}")


if __name__ == "__main__":
    main()
