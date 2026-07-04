"""Detect overlap-speech candidate regions from multi-channel VAD evidence.

The script marks regions where many channels have high VAD confidence at the
same time. It can also annotate those regions with boundary density, predicted
speaker switches, and reference-overlap coverage when RTTM files are available.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Detect overlap-speech candidate regions.")
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
        help="Directories containing NeMo vad_outputs/*.frame files, one directory per channel.",
    )
    parser.add_argument("--diar-rttm-dir", type=Path, default=None, help="Optional predicted RTTM directory.")
    parser.add_argument("--reference-rttm-dir", type=Path, default=None, help="Optional reference RTTM directory.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL file for overlap candidates.")
    parser.add_argument("--summary-output", type=Path, default=None, help="Optional summary JSON output.")
    parser.add_argument("--overlap-rttm-output-dir", type=Path, default=None, help="Optional RTTM marker directory.")
    parser.add_argument("--base-vad-manifest", type=Path, default=None, help="Optional base external VAD manifest.")
    parser.add_argument("--relaxed-vad-output", type=Path, default=None, help="Optional overlap-relaxed VAD manifest.")
    parser.add_argument("--vad-pattern", default="*.txt", help="VAD txt filename pattern.")
    parser.add_argument("--frame-pattern", default="*.frame", help="Frame filename pattern.")
    parser.add_argument("--frame-shift", type=float, default=0.01, help="Frame shift in seconds.")
    parser.add_argument("--score-threshold", type=float, default=0.5, help="Per-channel active threshold.")
    parser.add_argument("--min-active-channels", type=int, default=6, help="Minimum active channels for overlap evidence.")
    parser.add_argument("--min-mean-score", type=float, default=0.55, help="Minimum mean frame score.")
    parser.add_argument("--pad-onset", type=float, default=0.1, help="Pad candidate start in seconds.")
    parser.add_argument("--pad-offset", type=float, default=0.1, help="Pad candidate end in seconds.")
    parser.add_argument("--merge-gap", type=float, default=0.3, help="Merge candidate intervals separated by this gap.")
    parser.add_argument("--min-duration", type=float, default=0.2, help="Drop candidate intervals shorter than this.")
    parser.add_argument("--boundary-context", type=float, default=0.25, help="Context for VAD boundary counting.")
    parser.add_argument("--speaker-context", type=float, default=0.5, help="Context for predicted speaker switch counting.")
    parser.add_argument("--min-boundary-count", type=int, default=0, help="Optional minimum nearby VAD boundaries.")
    parser.add_argument("--min-speaker-switch-count", type=int, default=0, help="Optional minimum nearby speaker switches.")
    parser.add_argument(
        "--evidence-mode",
        choices=["and", "or"],
        default="and",
        help="How to combine boundary and speaker-switch filters when both are enabled.",
    )
    parser.add_argument("--relax-pad", type=float, default=0.15, help="Extra pad when writing relaxed VAD manifest.")
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


def read_rttm(path: Path):
    intervals = []
    if path is None or not path.is_file():
        return intervals

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            recording_id = parts[1]
            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            if duration > 0:
                intervals.append(
                    {
                        "recording_id": recording_id,
                        "start": start,
                        "end": start + duration,
                        "speaker": speaker,
                    }
                )
    return intervals


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


def detect_frame_candidates(channel_data, args):
    channel_indices = sorted(channel_data)
    scores_by_channel = [channel_data[idx]["scores"] for idx in channel_indices]
    min_frames = min(len(scores) for scores in scores_by_channel)
    if min_frames == 0:
        return []

    raw = []
    active_start = None
    for frame_idx in range(min_frames):
        frame_scores = [scores[frame_idx] for scores in scores_by_channel]
        active_channels = sum(score >= args.score_threshold for score in frame_scores)
        mean_score = sum(frame_scores) / len(frame_scores)
        is_overlap_like = active_channels >= args.min_active_channels and mean_score >= args.min_mean_score

        if is_overlap_like and active_start is None:
            active_start = frame_idx * args.frame_shift
        elif not is_overlap_like and active_start is not None:
            raw.append((active_start, frame_idx * args.frame_shift))
            active_start = None

    if active_start is not None:
        raw.append((active_start, min_frames * args.frame_shift))

    padded = pad_intervals(raw, args.pad_onset, args.pad_offset)
    return merge_intervals(padded, args.merge_gap, args.min_duration)


def frame_region_stats(channel_data, start, end, frame_shift, score_threshold):
    channel_indices = sorted(channel_data)
    scores_by_channel = [channel_data[idx]["scores"] for idx in channel_indices]
    min_frames = min(len(scores) for scores in scores_by_channel)
    start_idx = max(0, int(start / frame_shift))
    end_idx = min(min_frames, int(end / frame_shift) + 1)
    if end_idx <= start_idx:
        return {"mean_active_channels": 0.0, "max_active_channels": 0, "mean_frame_score": 0.0}

    active_sum = 0
    max_active = 0
    score_sum = 0.0
    frame_count = 0
    for frame_idx in range(start_idx, end_idx):
        frame_scores = [scores[frame_idx] for scores in scores_by_channel]
        active_channels = sum(score >= score_threshold for score in frame_scores)
        active_sum += active_channels
        max_active = max(max_active, active_channels)
        score_sum += sum(frame_scores) / len(frame_scores)
        frame_count += 1

    return {
        "mean_active_channels": active_sum / frame_count,
        "max_active_channels": max_active,
        "mean_frame_score": score_sum / frame_count,
    }


def vad_boundaries(channel_data):
    boundaries = []
    for data in channel_data.values():
        for start, end in data["intervals"]:
            boundaries.append(start)
            boundaries.append(end)
    return sorted(boundaries)


def count_values_in_region(values, start, end):
    return sum(start <= value <= end for value in values)


def speaker_switch_times(rttm_intervals):
    switches = []
    previous = None
    for item in sorted(rttm_intervals, key=lambda x: (x["start"], x["end"])):
        if previous is not None and item["speaker"] != previous["speaker"]:
            switches.append(item["start"])
        previous = item
    return switches


def overlap_duration(interval_a, interval_b):
    start = max(interval_a[0], interval_b[0])
    end = min(interval_a[1], interval_b[1])
    return max(0.0, end - start)


def intersection_duration(intervals_a, intervals_b):
    total = 0.0
    idx = 0
    intervals_b = sorted(intervals_b)
    for start_a, end_a in sorted(intervals_a):
        while idx < len(intervals_b) and intervals_b[idx][1] <= start_a:
            idx += 1
        cur = idx
        while cur < len(intervals_b) and intervals_b[cur][0] < end_a:
            total += overlap_duration((start_a, end_a), intervals_b[cur])
            cur += 1
    return total


def total_duration(intervals):
    return sum(max(0.0, end - start) for start, end in intervals)


def reference_overlap_regions(rttm_intervals):
    events = []
    for item in rttm_intervals:
        events.append((item["start"], 1))
        events.append((item["end"], -1))

    regions = []
    active = 0
    prev_time = None
    for time, delta in sorted(events):
        if prev_time is not None and time > prev_time and active >= 2:
            regions.append((prev_time, time))
        active += delta
        prev_time = time
    return merge_intervals(regions, merge_gap=0.0, min_duration=0.0)


def predicted_speaker_coverage(rttm_intervals, start, end):
    coverage = defaultdict(float)
    for item in rttm_intervals:
        dur = overlap_duration((start, end), (item["start"], item["end"]))
        if dur > 0:
            coverage[item["speaker"]] += dur
    return [
        {"speaker": speaker, "duration": round(duration, 4)}
        for speaker, duration in sorted(coverage.items(), key=lambda x: x[1], reverse=True)
    ]


def read_manifest(path: Path):
    by_recording = defaultdict(list)
    if path is None:
        return by_recording

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            recording_id = item.get("uniq_id") or Path(item["audio_filepath"]).stem
            start = float(item["offset"])
            end = start + float(item["duration"])
            by_recording[recording_id].append(
                {
                    "start": start,
                    "end": end,
                    "audio_filepath": item["audio_filepath"],
                    "label": item.get("label", "UNK"),
                }
            )
    return by_recording


def write_relaxed_manifest(base_manifest, candidates_by_recording, output: Path, relax_pad: float):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for recording_id in sorted(base_manifest):
            entries = base_manifest[recording_id]
            if not entries:
                continue
            audio_filepath = entries[0]["audio_filepath"]
            label = entries[0]["label"]
            intervals = [(item["start"], item["end"]) for item in entries]
            intervals.extend(
                (max(0.0, start - relax_pad), end + relax_pad)
                for start, end in candidates_by_recording.get(recording_id, [])
            )
            for start, end in merge_intervals(intervals, merge_gap=0.0, min_duration=0.0):
                json.dump(
                    {
                        "audio_filepath": audio_filepath,
                        "offset": round(start, 5),
                        "duration": round(end - start, 5),
                        "label": label,
                        "uniq_id": recording_id,
                    },
                    f,
                    ensure_ascii=False,
                )
                f.write("\n")


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator > 0 else 0.0


def passes_evidence_filters(boundary_count: int, speaker_switch_count: int, args):
    checks = []
    if args.min_boundary_count > 0:
        checks.append(boundary_count >= args.min_boundary_count)
    if args.min_speaker_switch_count > 0:
        checks.append(speaker_switch_count >= args.min_speaker_switch_count)
    if not checks:
        return True
    if args.evidence_mode == "or":
        return any(checks)
    return all(checks)


def main() -> None:
    args = parse_args()
    by_recording = collect_by_recording(args.vad_dirs, args.frame_dirs, args.vad_pattern, args.frame_pattern)
    if not by_recording:
        raise FileNotFoundError("No VAD/frame files found.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.overlap_rttm_output_dir is not None:
        args.overlap_rttm_output_dir.mkdir(parents=True, exist_ok=True)

    all_records = {}
    candidates_by_recording = {}
    summary = {"recordings": {}, "total": defaultdict(float)}

    with args.output.open("w", encoding="utf-8") as out:
        for recording_id in sorted(by_recording):
            channel_data = by_recording[recording_id]
            missing = [idx for idx, data in channel_data.items() if "intervals" not in data or "scores" not in data]
            if missing:
                raise ValueError(f"Missing VAD txt or frame scores for {recording_id}, channels: {missing}")

            raw_candidates = detect_frame_candidates(channel_data, args)
            pred_rttm = read_rttm(args.diar_rttm_dir / f"{recording_id}.rttm") if args.diar_rttm_dir else []
            ref_rttm = read_rttm(args.reference_rttm_dir / f"{recording_id}.rttm") if args.reference_rttm_dir else []
            ref_overlap = reference_overlap_regions(ref_rttm) if ref_rttm else []
            boundaries = vad_boundaries(channel_data)
            switches = speaker_switch_times(pred_rttm)

            if args.overlap_rttm_output_dir is not None:
                rttm_path = args.overlap_rttm_output_dir / f"{recording_id}.rttm"
                rttm_file = rttm_path.open("w", encoding="utf-8")
            else:
                rttm_file = None

            records = []
            kept_candidates = []
            for start, end in raw_candidates:
                stats = frame_region_stats(channel_data, start, end, args.frame_shift, args.score_threshold)
                boundary_start = max(0.0, start - args.boundary_context)
                boundary_end = end + args.boundary_context
                switch_start = max(0.0, start - args.speaker_context)
                switch_end = end + args.speaker_context
                boundary_count = count_values_in_region(boundaries, boundary_start, boundary_end)
                speaker_switch_count = count_values_in_region(switches, switch_start, switch_end)
                if not passes_evidence_filters(boundary_count, speaker_switch_count, args):
                    continue

                kept_candidates.append((start, end))
                ref_dur = intersection_duration([(start, end)], ref_overlap)
                record = {
                    "recording_id": recording_id,
                    "offset": round(start, 4),
                    "duration": round(end - start, 4),
                    "mean_active_channels": round(stats["mean_active_channels"], 4),
                    "max_active_channels": stats["max_active_channels"],
                    "mean_frame_score": round(stats["mean_frame_score"], 6),
                    "boundary_count": boundary_count,
                    "speaker_switch_count": speaker_switch_count,
                    "predicted_speakers": predicted_speaker_coverage(pred_rttm, start, end),
                    "reference_overlap_duration": round(ref_dur, 4),
                    "reference_overlap_ratio": round(safe_divide(ref_dur, end - start), 4),
                }
                records.append(record)
                json.dump(record, out, ensure_ascii=False)
                out.write("\n")

                if rttm_file is not None:
                    rttm_file.write(
                        f"SPEAKER {recording_id} 1 {start:.4f} {end - start:.4f} "
                        "<NA> <NA> overlap_candidate <NA> <NA>\n"
                    )

            if rttm_file is not None:
                rttm_file.close()

            candidates_by_recording[recording_id] = kept_candidates
            candidate_duration = total_duration(kept_candidates)
            ref_duration = total_duration(ref_overlap)
            intersect = intersection_duration(kept_candidates, ref_overlap)
            precision = safe_divide(intersect, candidate_duration)
            recall = safe_divide(intersect, ref_duration)
            f1 = safe_divide(2 * precision * recall, precision + recall)
            summary["recordings"][recording_id] = {
                "candidate_count": len(kept_candidates),
                "candidate_duration": round(candidate_duration, 4),
                "reference_overlap_duration": round(ref_duration, 4),
                "intersection_duration": round(intersect, 4),
                "duration_precision": round(precision, 6),
                "duration_recall": round(recall, 6),
                "duration_f1": round(f1, 6),
            }
            summary["total"]["candidate_count"] += len(kept_candidates)
            summary["total"]["candidate_duration"] += candidate_duration
            summary["total"]["reference_overlap_duration"] += ref_duration
            summary["total"]["intersection_duration"] += intersect
            all_records[recording_id] = records

    total_precision = safe_divide(summary["total"]["intersection_duration"], summary["total"]["candidate_duration"])
    total_recall = safe_divide(summary["total"]["intersection_duration"], summary["total"]["reference_overlap_duration"])
    total_f1 = safe_divide(2 * total_precision * total_recall, total_precision + total_recall)
    summary["total"] = {
        "candidate_count": int(summary["total"]["candidate_count"]),
        "candidate_duration": round(summary["total"]["candidate_duration"], 4),
        "reference_overlap_duration": round(summary["total"]["reference_overlap_duration"], 4),
        "intersection_duration": round(summary["total"]["intersection_duration"], 4),
        "duration_precision": round(total_precision, 6),
        "duration_recall": round(total_recall, 6),
        "duration_f1": round(total_f1, 6),
    }

    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_output.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    if args.base_vad_manifest is not None and args.relaxed_vad_output is not None:
        base_manifest = read_manifest(args.base_vad_manifest)
        write_relaxed_manifest(base_manifest, candidates_by_recording, args.relaxed_vad_output, args.relax_pad)

    print(
        "Wrote "
        f"{summary['total']['candidate_count']} overlap candidates "
        f"({summary['total']['candidate_duration']:.2f}s) to {args.output}"
    )
    if summary["total"]["reference_overlap_duration"] > 0:
        print(
            "Reference overlap analysis: "
            f"precision={summary['total']['duration_precision']:.4f}, "
            f"recall={summary['total']['duration_recall']:.4f}, "
            f"f1={summary['total']['duration_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
