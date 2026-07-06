#!/usr/bin/env python3
"""Split multi-channel FLAC → single-channel WAVs + run per-channel VAD.

Does NOT merge.  You can then use merge_vad_vote.py / merge_vad_weighted_vote.py
etc. to combine the per-channel VAD outputs however you like.

Usage
-----
    cd /home/wangy1/speech-diag-project
    source venv/bin/activate

    python scripts/split_and_vad.py \
        --audio data/wavs/L_R004S01C01.flac \
        --rttm  data/rttm/L_R004S01C01.rttm \
        --workdir results/mc_vad_L_R004S01C01 \
        --max-duration 300

    # Then fuse with whatever strategy you want, e.g.:
    python scripts/merge_vad_vote.py \
      --vad-dirs results/mc_vad_L_R004S01C01/vad_outputs/ch00 \
                ... ch07 \
      --audio-dir data/wavs_mono \
      --output data/manifests/aishell4_external_vad_vote2_pad02_one.json \
      --vote-threshold 2 --merge-gap 0.2 --min-duration 0.05 \
      --pad-onset 0.2 --pad-offset 0.2
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import soundfile as sf


PROJECT_ROOT = Path("/home/wangy1/speech-diag-project")
VAD_CONFIG_NAME = "diar_infer_vad"


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"[RUN] {' '.join(map(str, cmd))}", flush=True)
    return subprocess.run(cmd, check=True, **kwargs)


def split_channels(audio_path: Path, out_dir: Path, max_duration: float | None = None) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    data, sr = sf.read(str(audio_path))
    if data.ndim != 2:
        raise ValueError(f"Expected multi-channel audio, got shape {data.shape}")
    if max_duration is not None:
        max_samples = int(max_duration * sr)
        data = data[:max_samples, :]
        print(f"Truncating to {max_duration}s ({max_samples} samples)")
    n_ch = data.shape[1]
    print(f"Splitting {audio_path.name}: {data.shape} @ {sr} Hz → {n_ch} channels")

    channel_paths = []
    for ch in range(n_ch):
        ch_path = out_dir / f"{audio_path.stem}_ch{ch:02d}.wav"
        sf.write(str(ch_path), data[:, ch], sr)
        channel_paths.append(ch_path)
        print(f"  [{ch}] {ch_path}")
    return channel_paths


def write_channel_manifest(channel_path: Path, rttm_path: Path | None, manifest_path: Path, duration: float | None = None) -> None:
    entry = {
        "audio_filepath": str(channel_path.resolve()),
        "offset": 0,
        "duration": duration,
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


def run_vad_for_channel(manifest_path: Path, out_dir: Path, nemo_infer_dir: Path, channel_index: int) -> None:
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
    # After VAD, rename output txt files to strip _ch{idx} suffix
    # so merge_vad_vote.py can correctly group them as the same recording.
    vad_sub = out_dir / "vad_outputs"
    if vad_sub.is_dir():
        import re
        for txt in sorted(vad_sub.glob("*.txt")):
            new_name = re.sub(r'_ch\d{2}\.txt$', '.txt', txt.name)
            if new_name != txt.name:
                new_path = txt.with_name(new_name)
                txt.rename(new_path)
                print(f"    [rename] {txt.name} → {new_name}")


def main():
    p = argparse.ArgumentParser(description="Split multi-channel audio + run per-channel VAD")
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--rttm", type=Path, default=None)
    p.add_argument("--nemo-root", type=Path, default=PROJECT_ROOT / "baseline/NeMo")
    p.add_argument("--workdir", type=Path, default=PROJECT_ROOT / "results/mc_vad")
    p.add_argument("--skip-vad", action="store_true", help="Only split + manifests, skip VAD.")
    p.add_argument("--max-duration", type=float, default=None, help="Truncate audio to first N seconds before splitting/VAD.")
    args = p.parse_args()

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

    print("\n=== Step 1: Split channels ===")
    channel_paths = split_channels(args.audio, ch_wav_dir, args.max_duration)
    n_channels = len(channel_paths)

    print(f"\n=== Step 2: Generate {n_channels} manifests ===")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    rttm_resolved = args.rttm.resolve() if args.rttm else None
    for ch_idx, ch_path in enumerate(channel_paths):
        mf_path = manifest_dir / f"ch{ch_idx:02d}.json"
        write_channel_manifest(ch_path, rttm_resolved, mf_path, args.max_duration)

    if args.skip_vad:
        print("\n=== Step 3: Skipped (--skip-vad) ===")
    else:
        print(f"\n=== Step 3: Run VAD for {n_channels} channels (config: {VAD_CONFIG_NAME}.yaml) ===")
        for ch_idx, ch_path in enumerate(channel_paths):
            ch_name = f"ch{ch_idx:02d}"
            ch_out = vad_out_base / ch_name
            ch_manifest = manifest_dir / f"{ch_name}.json"
            print(f"\n--- Channel {ch_name} ---")
            run_vad_for_channel(ch_manifest, ch_out, nemo_infer_dir, ch_idx)

    print("\n" + "=" * 60)
    print("Split + VAD done!")
    print(f"  Channel WAVs  : {ch_wav_dir}")
    print(f"  Manifests     : {manifest_dir}")
    print(f"  VAD outputs   : {vad_out_base}")
    print()
    print("Now fuse with e.g.:")
    print(f"  python scripts/merge_vad_vote.py \\")
    for ch_idx in range(n_channels):
        print(f"    {vad_out_base}/ch{ch_idx:02d} \\")
    print(f"    --audio-dir data/wavs_mono \\")
    print(f"    --output data/manifests/aishell4_external_vad_{args.audio.stem}.json \\")
    print(f"    --vote-threshold 2 --merge-gap 0.2 --min-duration 0.05 --pad-onset 0.2 --pad-offset 0.2")
    print("=" * 60)


if __name__ == "__main__":
    main()
