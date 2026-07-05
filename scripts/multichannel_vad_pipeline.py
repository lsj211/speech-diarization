#!/usr/bin/env python3
"""End-to-end multi-channel VAD union pipeline for NeMo clustering diarizer.

Steps:
  1. Split an 8-channel FLAC into 8 single-channel WAVs.
  2. Generate per-channel NeMo manifests.
  3. Run vad_only_infer.py for each channel using the shared diar_infer_vad.yaml.
  4. Merge the 8 VAD outputs by taking the temporal union.
  5. (Optional) Run diarization with the merged external VAD manifest.

The shared VAD-only YAML config lives at:
    baseline/NeMo/examples/speaker_tasks/diarization/conf/inference/diar_infer_vad.yaml

Usage
-----
    cd /home/wangy1/speech-diag-project
    source venv/bin/activate

    # Full pipeline on one recording:
    python scripts/multichannel_vad_pipeline.py \
        --audio data/wavs/L_R004S01C01.flac \
        --rttm  data/rttm/L_R004S01C01.rttm \
        --workdir results/mc_vad_L_R004S01C01

    # Then run diarization with the merged VAD:
    cd baseline/NeMo/examples/speaker_tasks/diarization/clustering_diarizer
    python -u offline_diar_infer.py \
        --config-name=diar_infer_local \
        device=cpu \
        diarizer.vad.external_vad_manifest=<workdir>/merged_vad.json \
        diarizer.vad.model_path=null
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


PROJECT_ROOT = Path("/home/wangy1/speech-diag-project")

# Path to the shared VAD-only YAML (relative to NeMo conf dir)
# vad_only_infer.py loads from "../conf/inference" relative to its own directory,
# so config-name="diar_infer_vad" resolves to this file.
VAD_CONFIG_NAME = "diar_infer_vad"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"[RUN] {' '.join(map(str, cmd))}", flush=True)
    return subprocess.run(cmd, check=True, **kwargs)


# ---------------------------------------------------------------------------
# Step 1: split 8-channel FLAC → 8 single-channel WAVs
# ---------------------------------------------------------------------------


def split_channels(audio_path: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    data, sr = sf.read(str(audio_path))
    if data.ndim != 2:
        raise ValueError(f"Expected multi-channel audio, got shape {data.shape}")
    n_ch = data.shape[1]
    print(f"Splitting {audio_path.name}: {data.shape} @ {sr} Hz → {n_ch} channels")

    channel_paths = []
    for ch in range(n_ch):
        ch_path = out_dir / f"{audio_path.stem}_ch{ch:02d}.wav"
        sf.write(str(ch_path), data[:, ch], sr)
        channel_paths.append(ch_path)
        print(f"  [{ch}] {ch_path}")
    return channel_paths


# ---------------------------------------------------------------------------
# Step 2: generate per-channel manifest JSONL
# ---------------------------------------------------------------------------


def write_channel_manifest(channel_path: Path, rttm_path: Path | None, manifest_path: Path) -> None:
    entry = {
        "audio_filepath": str(channel_path.resolve()),
        "offset": 0,
        "duration": None,
        "label": "infer",
        "text": "-",
        "num_speakers": None,
        "rttm_filepath": str(rttm_path.resolve()) if rttm_path else None,
        "uem_filepath": None,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(entry, f, ensure_ascii=False)
        f.write("\n")
    print(f"  manifest → {manifest_path}")


# ---------------------------------------------------------------------------
# Step 3: patch the shared VAD YAML with channel-specific values and run VAD
# ---------------------------------------------------------------------------


def run_vad_for_channel(
    manifest_path: Path,
    out_dir: Path,
    nemo_infer_dir: Path,
) -> None:
    """Run vad_only_infer.py for a single channel.

    Overrides diarizer.manifest_filepath and diarizer.out_dir via Hydra CLI
    so we never mutate the shared YAML on disk.
    """
    run(
        [
            sys.executable, "-u",
            str(nemo_infer_dir / "vad_only_infer.py"),
            f"--config-name={VAD_CONFIG_NAME}",
            "device=cpu",
            f"diarizer.manifest_filepath={manifest_path.resolve()}",
            f"diarizer.out_dir={out_dir.resolve()}",
        ],
        cwd=str(nemo_infer_dir),
    )


# ---------------------------------------------------------------------------
# Step 4: merge VAD .txt outputs → union external VAD manifest
# ---------------------------------------------------------------------------


def read_vad_txt(path: Path) -> list[tuple[float, float]]:
    intervals = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            start = float(parts[0])
            dur = float(parts[1])
            if dur > 0:
                intervals.append((start, start + dur))
    return intervals


def merge_intervals(
    intervals: list[tuple[float, float]],
    merge_gap: float = 0.2,
    min_duration: float = 0.05,
) -> list[tuple[float, float]]:
    if not intervals:
        return []
    sorted_iv = sorted(intervals)
    merged = [sorted_iv[0]]
    for start, end in sorted_iv[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + merge_gap:
            merged[-1] = (last_start, max(last_end, end))
        else:
            if last_end - last_start >= min_duration:
                merged[-1] = (last_start, last_end)
            else:
                merged.pop()
            merged.append((start, end))
    if merged and merged[-1][1] - merged[-1][0] < min_duration:
        merged.pop()
    return merged


def merge_vad_outputs(
    vad_out_dirs: list[Path],
    audio_filepath: str,
    output_path: Path,
    merge_gap: float = 0.2,
    min_duration: float = 0.05,
) -> None:
    all_intervals = []
    for vad_dir in vad_out_dirs:
        for txt_file in sorted(vad_dir.glob("*.txt")):
            all_intervals.extend(read_vad_txt(txt_file))

    print(f"Total raw VAD intervals across {len(vad_out_dirs)} channels: {len(all_intervals)}")
    merged = merge_intervals(all_intervals, merge_gap, min_duration)
    print(f"After union merge: {len(merged)} intervals")

    recording_id = Path(audio_filepath).stem
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for start, end in merged:
            entry = {
                "audio_filepath": audio_filepath,
                "offset": round(start, 5),
                "duration": round(end - start, 5),
                "label": "UNK",
                "uniq_id": recording_id,
            }
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

    total_speech = sum(e - s for s, e in merged)
    print(f"Merged VAD manifest → {output_path}  ({total_speech:.1f}s speech)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-channel VAD union pipeline for NeMo clustering diarizer",
    )
    p.add_argument("--audio", type=Path, required=True, help="Multi-channel FLAC file.")
    p.add_argument(
        "--rttm", type=Path, default=None, help="Optional reference RTTM (not used by VAD, only forwarded)."
    )
    p.add_argument(
        "--nemo-root",
        type=Path,
        default=PROJECT_ROOT / "baseline/NeMo",
        help="NeMo git root.",
    )
    p.add_argument(
        "--workdir",
        type=Path,
        default=PROJECT_ROOT / "results/mc_vad",
        help="Working directory for channel WAVs, manifests, VAD outputs.",
    )
    p.add_argument("--merge-gap", type=float, default=0.2, help="Merge VAD intervals ≤ this gap apart.")
    p.add_argument("--min-duration", type=float, default=0.05, help="Drop merged intervals shorter than this.")
    p.add_argument(
        "--skip-vad", action="store_true", help="Skip running VAD (only do split + merge from existing outputs)."
    )
    return p.parse_args()


def main():
    args = parse_args()
    nemo_root = args.nemo_root.resolve()
    nemo_infer_dir = nemo_root / "examples/speaker_tasks/diarization/clustering_diarizer"

    if not args.audio.exists():
        sys.exit(f"Audio file not found: {args.audio}")
    if not nemo_infer_dir.is_dir():
        sys.exit(f"NeMo inference dir not found: {nemo_infer_dir}")

    workdir = args.workdir.resolve()
    ch_wav_dir = workdir / "channels"
    manifest_dir = workdir / "manifests"
    vad_out_base = workdir / "vad_outputs"
    merged_vad_path = workdir / "merged_vad.json"

    # ── Step 1: split channels ──────────────────────────────────────
    print("\n=== Step 1: Split channels ===")
    channel_paths = split_channels(args.audio, ch_wav_dir)
    n_channels = len(channel_paths)

    # ── Step 2: per-channel manifests ────────────────────────────────
    print(f"\n=== Step 2: Generate {n_channels} manifests ===")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for ch_idx, ch_path in enumerate(channel_paths):
        mf_path = manifest_dir / f"ch{ch_idx:02d}.json"
        write_channel_manifest(ch_path, args.rttm.resolve() if args.rttm else None, mf_path)

    # ── Step 3: run VAD per channel ──────────────────────────────────
    if not args.skip_vad:
        print(f"\n=== Step 3: Run VAD for {n_channels} channels (config: {VAD_CONFIG_NAME}.yaml) ===")
        for ch_idx, ch_path in enumerate(channel_paths):
            ch_name = f"ch{ch_idx:02d}"
            ch_out = vad_out_base / ch_name
            ch_manifest = manifest_dir / f"{ch_name}.json"
            print(f"\n--- Channel {ch_name} ---")
            run_vad_for_channel(ch_manifest, ch_out, nemo_infer_dir)
    else:
        print("\n=== Step 3: Skipped (--skip-vad) ===")

    # ── Step 4: merge VAD outputs ────────────────────────────────────
    print("\n=== Step 4: Merge VAD outputs (union) ===")
    vad_dirs = sorted(vad_out_base.glob("ch*"))
    if not vad_dirs:
        sys.exit(f"No VAD output directories found under {vad_out_base}")
    mono_audio = str(PROJECT_ROOT / "data/wavs_mono" / args.audio.name)
    merge_vad_outputs(vad_dirs, mono_audio, merged_vad_path, args.merge_gap, args.min_duration)

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print(f"  Merged VAD manifest: {merged_vad_path}")
    print()
    print("Next – run diarization with merged VAD:")
    print(f"  cd {nemo_infer_dir}")
    print(f"  python -u offline_diar_infer.py \\")
    print(f"      --config-name=diar_infer_local \\")
    print(f"      device=cpu \\")
    print(f"      diarizer.vad.external_vad_manifest={merged_vad_path} \\")
    print(f"      diarizer.vad.model_path=null")
    print("=" * 60)


if __name__ == "__main__":
    main()
