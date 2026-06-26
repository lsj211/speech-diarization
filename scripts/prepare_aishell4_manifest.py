"""Prepare NeMo diarization manifests for AISHELL-4.

By default, this script reads the project-level flat layout:

    data/wavs/*.flac
    data/rttm/*.rttm

It can also read the original AISHELL-4 split layout with --data-root:

    <data-root>/wav/*.flac
    <data-root>/TextGrid/*.rttm
"""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Create NeMo manifest files for AISHELL-4 diarization data.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Optional AISHELL-4 split directory containing wav/ and TextGrid/.",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path("data/wavs"),
        help="Directory containing audio files. Ignored when --data-root is set.",
    )
    parser.add_argument(
        "--rttm-dir",
        type=Path,
        default=Path("data/rttm"),
        help="Directory containing RTTM files. Ignored when --data-root is set.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/aishell4_test_manifest.json"),
        help="Output JSONL manifest path.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optionally keep only the first N sessions.")
    parser.add_argument(
        "--with-num-speakers",
        action="store_true",
        help="Fill num_speakers from RTTM speaker labels. Otherwise write null.",
    )
    return parser.parse_args()


def count_speakers(rttm_path: Path) -> int:
    speakers = set()
    with rttm_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == "SPEAKER":
                speakers.add(parts[7])
    return len(speakers)


def main() -> None:
    args = parse_args()
    if args.data_root is not None:
        audio_dir = args.data_root / "wav"
        rttm_dir = args.data_root / "TextGrid"
    else:
        audio_dir = args.audio_dir
        rttm_dir = args.rttm_dir

    if not audio_dir.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    if not rttm_dir.is_dir():
        raise FileNotFoundError(f"RTTM directory not found: {rttm_dir}")

    audio_paths = sorted(audio_dir.glob("*.flac"))
    if args.limit is not None:
        audio_paths = audio_paths[: args.limit]

    entries = []
    missing = []
    for audio_path in audio_paths:
        rttm_path = rttm_dir / f"{audio_path.stem}.rttm"
        if not rttm_path.is_file():
            missing.append(rttm_path)
            continue
        entry = {
            "audio_filepath": str(audio_path.resolve()).replace("\\", "/"),
            "offset": 0,
            "duration": None,
            "label": "infer",
            "text": "-",
            "num_speakers": count_speakers(rttm_path) if args.with_num_speakers else None,
            "rttm_filepath": str(rttm_path.resolve()).replace("\\", "/"),
            "uem_filepath": None,
        }
        entries.append(entry)

    if missing:
        missing_text = "\n".join(str(path) for path in missing[:10])
        raise FileNotFoundError(f"Missing RTTM files for {len(missing)} audio files:\n{missing_text}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for entry in entries:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

    print(f"Wrote {len(entries)} entries to {args.output}")


if __name__ == "__main__":
    main()
