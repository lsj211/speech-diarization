"""Fuse multi-channel NeMo VAD outputs with SNR-weighted voting.

Like merge_vad_vote.py, but instead of hard k-of-n voting, each channel
contributes a soft weight derived from its estimated SNR:

    SNR_i = 10 * log10( power(speech_frames_i) / power(non_speech_frames_i) )
    w_i   = softmax(SNR_i / temperature)

Speech is kept where  Σ w_i >= weight_threshold * Σ w_i  (default 0.25).

Input format (same as merge_vad_vote.py):
    NeMo VAD txt:  <start_sec> <duration_sec> speech

Output: JSONL manifest for diarizer.vad.external_vad_manifest.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fuse multi-channel NeMo VAD outputs with SNR-weighted voting."
    )
    parser.add_argument(
        "--vad-dirs",
        nargs="+",
        type=Path,
        required=True,
        help="Directories containing NeMo vad_outputs/*.txt files, one directory per channel.",
    )
    parser.add_argument(
        "--channels-dir",
        type=Path,
        required=True,
        help="Directory containing per-channel mono WAV files used to estimate SNR.",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        required=True,
        help="Mono audio directory used for embedding extraction in the external manifest.",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output external VAD JSONL manifest."
    )
    parser.add_argument("--pattern", default="*.txt", help="VAD txt filename pattern.")
    parser.add_argument(
        "--weight-threshold",
        type=float,
        default=0.25,
        help="Normalised weight threshold for speech (0-1). Speech is kept "
        "when sum of active-channel weights >= this fraction of total weight.",
    )
    parser.add_argument(
        "--snr-temperature",
        type=float,
        default=2.0,
        help="Temperature for softmax over SNR values. Lower -> more contrast between channels.",
    )
    parser.add_argument(
        "--merge-gap",
        type=float,
        default=0.2,
        help="Merge speech intervals separated by this gap.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.05,
        help="Drop merged intervals shorter than this.",
    )
    parser.add_argument(
        "--pre-pad-min-duration",
        type=float,
        default=0.0,
        help="Drop weighted intervals shorter than this before padding.",
    )
    parser.add_argument(
        "--pad-onset",
        type=float,
        default=0.0,
        help="Add this many seconds before each speech segment.",
    )
    parser.add_argument(
        "--pad-offset",
        type=float,
        default=0.0,
        help="Add this many seconds after each speech segment.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# VAD txt I/O  (identical to merge_vad_vote.py)
# ---------------------------------------------------------------------------

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


def collect_recordings(vad_dirs, pattern):
    """Same semantics as merge_vad_vote.py: recording_id -> list of per-channel intervals."""
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


# ---------------------------------------------------------------------------
# SNR estimation
# ---------------------------------------------------------------------------

def estimate_snr(audio: np.ndarray, sample_rate: int, vad_intervals: list):
    """Estimate channel SNR using VAD intervals to label speech vs. non-speech.

    Returns SNR in dB.  Returns -inf when speech or noise power is too low.
    """
    if len(audio) == 0 or len(vad_intervals) == 0:
        return -float("inf")

    speech_mask = np.zeros(len(audio), dtype=bool)
    for start, end in vad_intervals:
        s = max(0, int(start * sample_rate))
        e = min(len(audio), int(end * sample_rate))
        if e > s:
            speech_mask[s:e] = True

    speech_power = float(np.mean(audio[speech_mask] ** 2)) if speech_mask.any() else 0.0
    noise_power = (
        float(np.mean(audio[~speech_mask] ** 2)) if (~speech_mask).any() else 1e-10
    )

    if noise_power < 1e-10 or speech_power < 1e-10:
        return -float("inf")

    return 10.0 * np.log10(speech_power / noise_power)


def snr_to_weights(snr_values: list, temperature: float = 2.0):
    """Softmax-normalised channel weights from SNR (dB).

    Channels with -inf SNR get weight 0.
    """
    arr = np.array(snr_values, dtype=np.float64)
    finite = np.isfinite(arr)

    if not finite.any():
        # All channels have invalid SNR -> uniform weights
        return np.ones(len(arr)) / len(arr)

    # Replace -inf with the minimum finite SNR
    min_finite = arr[finite].min()
    arr[~finite] = min_finite

    # Softmax with temperature
    arr_shifted = arr - arr.max()  # for numerical stability
    exp_arr = np.exp(arr_shifted / max(temperature, 0.01))
    weights = exp_arr / exp_arr.sum()

    # Zero out channels that originally had -inf SNR
    weights[~finite] = 0.0
    if weights.sum() < 1e-10:
        return np.ones(len(arr)) / len(arr)

    return weights / weights.sum()


# ---------------------------------------------------------------------------
# Weighted voting + post-processing
# ---------------------------------------------------------------------------

def weighted_vote_intervals(channel_intervals, weights, weight_threshold=0.25):
    """Sweep-line voting where each channel contributes its SNR weight.

    Speech is kept when  sum(active weights) >= weight_threshold * sum(all weights).
    """
    total_weight = sum(weights)
    if total_weight <= 0:
        return []

    threshold = weight_threshold * total_weight

    events = defaultdict(float)
    for intervals, w in zip(channel_intervals, weights):
        if w <= 0:
            continue
        for start, end in intervals:
            if end > start:
                events[start] += w
                events[end] -= w

    if not events:
        return []

    voted = []
    active = 0.0
    prev_t = None
    for t in sorted(events):
        if prev_t is not None and t > prev_t and active >= threshold:
            voted.append((prev_t, t))
        active += events[t]
        prev_t = t
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
    return [(s, e) for s, e in intervals if e - s >= min_duration]


def pad_intervals(intervals, pad_onset: float, pad_offset: float):
    if pad_onset <= 0 and pad_offset <= 0:
        return intervals
    return [(max(0.0, s - pad_onset), e + pad_offset) for s, e in intervals]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not (0 < args.weight_threshold <= 1):
        raise ValueError("--weight-threshold must be in (0, 1]")

    vad_by_recording = collect_recordings(args.vad_dirs, args.pattern)
    if not vad_by_recording:
        raise FileNotFoundError("No VAD txt files found.")

    num_channels = len(args.vad_dirs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_segments = 0

    with args.output.open("w", encoding="utf-8") as f:
        for recording_id in sorted(vad_by_recording):
            channel_vads = vad_by_recording[recording_id]
            if len(channel_vads) != num_channels:
                print(
                    f"[WARN] {recording_id}: {len(channel_vads)} VAD sets but "
                    f"{num_channels} channels -- skipping"
                )
                continue

            # ---- SNR estimation per channel ----
            snr_values = []
            for ch_idx, vad_intervals in enumerate(channel_vads):
                wav_path = args.channels_dir / f"{recording_id}_ch{ch_idx:02d}.wav"
                if not wav_path.is_file():
                    print(f"[WARN] {recording_id}: channel audio missing {wav_path}")
                    snr_values.append(-float("inf"))
                    continue

                audio, sr = sf.read(str(wav_path))
                if audio.ndim > 1:
                    audio = audio[:, 0]
                snr = estimate_snr(audio, sr, vad_intervals)
                snr_values.append(snr)

            weights = snr_to_weights(snr_values, args.snr_temperature)

            # Print SNR / weight summary for this recording
            snr_str = ", ".join(
                f"ch{i}={v:.1f}" for i, v in enumerate(snr_values) if np.isfinite(v)
            )
            w_str = ", ".join(f"ch{i}={w:.3f}" for i, w in enumerate(weights))
            print(f"[{recording_id}] SNR(dB): {snr_str}")
            print(f"[{recording_id}] weights: {w_str}")

            # ---- Weighted voting ----
            voted = weighted_vote_intervals(
                channel_vads, weights, args.weight_threshold
            )
            voted = filter_intervals(voted, args.pre_pad_min_duration)
            padded = pad_intervals(voted, args.pad_onset, args.pad_offset)
            merged = merge_intervals(padded, args.merge_gap, args.min_duration)

            # ---- Resolve mono audio path for output manifest ----
            base_id = recording_id
            for ch in range(num_channels * 2):  # generous upper bound
                suffix = f"_ch{ch:02d}"
                if base_id.endswith(suffix):
                    base_id = base_id[: -len(suffix)]
                    break

            audio_path = args.audio_dir / f"{base_id}.flac"
            if not audio_path.is_file():
                raise FileNotFoundError(
                    f"Audio file not found for {base_id}: {audio_path}"
                )

            for start, end in merged:
                entry = {
                    "audio_filepath": str(audio_path.resolve()).replace("\\", "/"),
                    "offset": round(start, 5),
                    "duration": round(end - start, 5),
                    "label": "UNK",
                    "uniq_id": base_id,
                }
                json.dump(entry, f, ensure_ascii=False)
                f.write("\n")
            total_segments += len(merged)

    print(
        f"Wrote {total_segments} SNR-weighted VAD segments "
        f"for {len(vad_by_recording)} recordings "
        f"to {args.output} "
        f"(weight-threshold={args.weight_threshold}, temperature={args.snr_temperature})"
    )


if __name__ == "__main__":
    main()
