"""Prepare chunked AISHELL-4 manifests for Sortformer fine-tuning.

The downloaded train_L/train_M layout is expected to be one or more pairs like:

    train_L/train_L/wav/*.flac
    train_L/train_L/TextGrid/*.rttm
    train_M/wav/*.flac
    train_M/TextGrid/*.rttm

This script does not convert audio to mono. Run scripts/convert_to_mono.py first
if you want the training audio to match the current test-set preprocessing.
"""

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SessionInfo:
    session_id: str
    audio_path: Path
    rttm_path: Path
    num_speakers: int
    duration: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create train/dev manifests for Sortformer 8-speaker fine-tuning.")
    parser.add_argument("--audio-dir", type=Path, nargs="+", default=[Path("train_L/train_L/wav")])
    parser.add_argument("--rttm-dir", type=Path, nargs="+", default=[Path("train_L/train_L/TextGrid")])
    parser.add_argument("--output-dir", type=Path, default=Path("data/manifests/model_training"))
    parser.add_argument("--prefix", default="aishell4_train_L_sortformer")
    parser.add_argument("--max-sessions", type=int, default=30)
    parser.add_argument("--dev-sessions", type=int, default=5)
    parser.add_argument("--min-speakers", type=int, default=5)
    parser.add_argument("--max-speakers", type=int, default=8)
    parser.add_argument("--chunk-sec", type=float, default=120.0)
    parser.add_argument("--chunk-hop-sec", type=float, default=120.0)
    parser.add_argument("--min-speech-sec", type=float, default=5.0)
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Write absolute paths. By default paths are relative to the manifest directory.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_rttm_segments(rttm_path: Path) -> list[tuple[float, float, str]]:
    segments: list[tuple[float, float, str]] = []
    with rttm_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            start = float(parts[3])
            duration = float(parts[4])
            if duration <= 0:
                continue
            segments.append((start, start + duration, parts[7]))
    return segments


def speech_seconds_in_window(segments: Iterable[tuple[float, float, str]], start: float, end: float) -> float:
    total = 0.0
    for seg_start, seg_end, _ in segments:
        overlap = min(seg_end, end) - max(seg_start, start)
        if overlap > 0:
            total += overlap
    return total


def collect_sessions(args: argparse.Namespace) -> list[SessionInfo]:
    if len(args.audio_dir) != len(args.rttm_dir):
        raise ValueError("--audio-dir and --rttm-dir must provide the same number of directories")

    sessions: list[SessionInfo] = []
    seen_session_ids: set[str] = set()
    for audio_dir, rttm_dir in zip(args.audio_dir, args.rttm_dir):
        if not audio_dir.is_dir():
            raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
        if not rttm_dir.is_dir():
            raise FileNotFoundError(f"RTTM directory not found: {rttm_dir}")

        for audio_path in sorted(audio_dir.glob("*.flac")):
            rttm_path = rttm_dir / f"{audio_path.stem}.rttm"
            if not rttm_path.is_file():
                raise FileNotFoundError(f"Missing RTTM for {audio_path.name}: {rttm_path}")
            if audio_path.stem in seen_session_ids:
                raise ValueError(f"Duplicate session id across inputs: {audio_path.stem}")
            seen_session_ids.add(audio_path.stem)

            segments = read_rttm_segments(rttm_path)
            speakers = {speaker for _, _, speaker in segments}
            duration = max((end for _, end, _ in segments), default=0.0)
            if args.min_speakers <= len(speakers) <= args.max_speakers:
                sessions.append(
                    SessionInfo(
                        session_id=audio_path.stem,
                        audio_path=audio_path.resolve(),
                        rttm_path=rttm_path.resolve(),
                        num_speakers=len(speakers),
                        duration=round(duration, 3),
                    )
                )

    rng = random.Random(args.seed)
    rng.shuffle(sessions)
    return sorted(sessions[: args.max_sessions], key=lambda item: item.session_id)


def split_sessions(sessions: list[SessionInfo], dev_sessions: int, seed: int) -> tuple[list[SessionInfo], list[SessionInfo]]:
    if dev_sessions <= 0:
        return sessions, []
    if dev_sessions >= len(sessions):
        raise ValueError("--dev-sessions must be smaller than the selected session count")

    rng = random.Random(seed)
    shuffled = sessions[:]
    rng.shuffle(shuffled)
    dev_ids = {item.session_id for item in shuffled[:dev_sessions]}
    train = [item for item in sessions if item.session_id not in dev_ids]
    dev = [item for item in sessions if item.session_id in dev_ids]
    return train, dev


def format_manifest_path(path: Path, manifest_dir: Path, absolute: bool) -> str:
    if absolute:
        return str(path.resolve()).replace("\\", "/")
    return os.path.relpath(path.resolve(), manifest_dir.resolve()).replace("\\", "/")


def build_chunk_entries(
    sessions: list[SessionInfo],
    manifest_dir: Path,
    chunk_sec: float,
    hop_sec: float,
    min_speech_sec: float,
    absolute_paths: bool,
) -> list[dict]:
    entries: list[dict] = []
    for session in sessions:
        segments = read_rttm_segments(session.rttm_path)
        audio_filepath = format_manifest_path(session.audio_path, manifest_dir, absolute_paths)
        rttm_filepath = format_manifest_path(session.rttm_path, manifest_dir, absolute_paths)
        offset = 0.0
        while offset < session.duration:
            duration = min(chunk_sec, session.duration - offset)
            if duration <= 0:
                break
            speech_sec = speech_seconds_in_window(segments, offset, offset + duration)
            if speech_sec >= min_speech_sec:
                entries.append(
                    {
                        "audio_filepath": audio_filepath,
                        "offset": round(offset, 3),
                        "duration": round(duration, 3),
                        "label": "infer",
                        "text": "-",
                        "num_speakers": session.num_speakers,
                        "rttm_filepath": rttm_filepath,
                        "uem_filepath": None,
                        "session_id": session.session_id,
                    }
                )
            offset += hop_sec
    return entries


def write_jsonl(path: Path, entries: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sessions = collect_sessions(args)
    if not sessions:
        raise RuntimeError("No usable sessions found after speaker-count filtering.")

    train_sessions, dev_sessions = split_sessions(sessions, args.dev_sessions, args.seed)
    train_manifest = args.output_dir / f"{args.prefix}_train.json"
    dev_manifest = args.output_dir / f"{args.prefix}_dev.json"
    session_manifest = args.output_dir / f"{args.prefix}_sessions.json"
    train_entries = build_chunk_entries(
        train_sessions,
        train_manifest.parent,
        args.chunk_sec,
        args.chunk_hop_sec,
        args.min_speech_sec,
        args.absolute_paths,
    )
    dev_entries = build_chunk_entries(
        dev_sessions,
        dev_manifest.parent,
        args.chunk_sec,
        args.chunk_hop_sec,
        args.min_speech_sec,
        args.absolute_paths,
    )

    session_entries = [
        {
            **asdict(item),
            "audio_path": format_manifest_path(item.audio_path, session_manifest.parent, args.absolute_paths),
            "rttm_path": format_manifest_path(item.rttm_path, session_manifest.parent, args.absolute_paths),
        }
        for item in sessions
    ]

    write_jsonl(train_manifest, train_entries)
    write_jsonl(dev_manifest, dev_entries)
    write_jsonl(session_manifest, session_entries)

    speaker_hist: dict[int, int] = {}
    for item in sessions:
        speaker_hist[item.num_speakers] = speaker_hist.get(item.num_speakers, 0) + 1

    print(f"Selected sessions: {len(sessions)} | speaker histogram: {speaker_hist}")
    print(f"Train sessions: {len(train_sessions)} | chunks: {len(train_entries)} | {train_manifest}")
    print(f"Dev sessions:   {len(dev_sessions)} | chunks: {len(dev_entries)} | {dev_manifest}")
    print(f"Session list: {session_manifest}")


if __name__ == "__main__":
    main()
