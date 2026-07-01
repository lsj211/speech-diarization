"""Convert AISHELL-4 audio files to mono FLAC files.

The original AISHELL-4 meeting recordings can be multi-channel. NeMo's
clustering diarizer VAD path expects mono waveforms, so this script averages
channels and writes single-channel audio while preserving the sample rate.
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


def parse_args():
    parser = argparse.ArgumentParser(description="Convert audio files to mono for NeMo diarization.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/wavs"), help="Directory with input audio files.")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/wavs_mono"), help="Directory for mono output files."
    )
    parser.add_argument("--pattern", default="*.flac", help="Input filename glob pattern.")
    parser.add_argument(
        "--channel",
        default="average",
        help="Use 'average' to average all channels, or an integer channel index such as 0.",
    )
    return parser.parse_args()


def to_mono(audio: np.ndarray, channel: str) -> np.ndarray:
    if audio.ndim == 1:
        return audio

    if channel == "average":
        return audio.mean(axis=1)

    channel_idx = int(channel)
    return audio[:, channel_idx]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    audio_paths = sorted(args.input_dir.glob(args.pattern))
    if not audio_paths:
        raise FileNotFoundError(f"No files matched {args.input_dir / args.pattern}")

    for audio_path in audio_paths:
        audio, sample_rate = sf.read(audio_path, always_2d=False)
        mono = to_mono(audio, args.channel)
        output_path = args.output_dir / audio_path.name
        sf.write(output_path, mono, sample_rate)
        print(f"{audio_path.name}: {audio.shape} -> {mono.shape}, {sample_rate} Hz")

    print(f"Wrote {len(audio_paths)} mono files to {args.output_dir}")


if __name__ == "__main__":
    main()

