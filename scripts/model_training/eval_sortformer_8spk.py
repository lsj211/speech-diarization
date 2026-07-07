"""Evaluate an 8-speaker Sortformer checkpoint on chunked diarization manifests.

This script is intentionally separate from NeMo's original e2e example because
the original path assumes a 4-speaker Sortformer in a few places and can also
OOM on long AISHELL-4 meetings. Use a chunked manifest for evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NEMO_ROOT = PROJECT_ROOT / "baseline" / "NeMo"
E2E_ROOT = NEMO_ROOT / "examples" / "speaker_tasks" / "diarization" / "neural_diarizer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned 8-speaker Sortformer.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("results/model_training/sortformer_8spk/sortformer_8spk_aishell4_train_L.nemo"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/model_training/aishell4_test_sortformer_chunks_train.json"),
    )
    parser.add_argument(
        "--out-rttm-dir",
        type=Path,
        default=Path("results/model_training/sortformer_8spk/eval_rttm_chunks"),
    )
    parser.add_argument("--max-speakers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--collar", type=float, default=0.25)
    parser.add_argument("--onset", type=float, default=None, help="Speech onset threshold for post-processing.")
    parser.add_argument("--offset", type=float, default=None, help="Speech offset threshold for post-processing.")
    parser.add_argument("--ignore-overlap", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bypass-postprocessing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--postprocessing-yaml", type=Path, default=None)
    parser.add_argument("--verbose-report", action="store_true", help="Print NeMo's per-chunk DER table.")
    return parser.parse_args()


def import_nemo_deps():
    for path in (str(E2E_ROOT), str(NEMO_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)

    import torch
    from omegaconf import OmegaConf
    from tqdm import tqdm

    from e2e_diarize_speech import convert_pred_mat_to_segments
    from nemo.collections.asr.metrics.der import score_labels
    from nemo.collections.asr.models import SortformerEncLabelModel
    from nemo.collections.asr.parts.utils.speaker_utils import audio_rttm_map
    from nemo.collections.asr.parts.utils.vad_utils import load_postprocessing_from_yaml

    return {
        "torch": torch,
        "OmegaConf": OmegaConf,
        "tqdm": tqdm,
        "convert_pred_mat_to_segments": convert_pred_mat_to_segments,
        "score_labels": score_labels,
        "SortformerEncLabelModel": SortformerEncLabelModel,
        "audio_rttm_map": audio_rttm_map,
        "load_postprocessing_from_yaml": load_postprocessing_from_yaml,
    }


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def chunk_uniq_id(item: dict, index: int) -> str:
    if item.get("uniq_id"):
        return str(item["uniq_id"])

    rttm_path = item.get("rttm_filepath") or item["audio_filepath"]
    base = Path(rttm_path).stem
    offset_ms = int(round(float(item.get("offset") or 0.0) * 1000))
    duration = item.get("duration")
    if duration is None:
        end_ms = offset_ms
    else:
        end_ms = int(round((float(item.get("offset") or 0.0) + float(duration)) * 1000))
    return f"{base}_chunk{index:04d}_{offset_ms:09d}_{end_ms:09d}"


def resolve_manifest_path(value: str | None, manifest: Path) -> str | None:
    if value in (None, ""):
        return value

    path = Path(value)
    candidates = [path] if path.is_absolute() else [manifest.parent / path, PROJECT_ROOT / path]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve()).replace("\\", "/")
    return str(candidates[0].resolve()).replace("\\", "/")


def write_manifest_with_unique_ids(manifest: Path) -> Path:
    """Create a sibling manifest with stable uniq_id fields for every chunk."""
    output_manifest = manifest.with_name(f"{manifest.stem}_uniq.json")
    used_ids: set[str] = set()

    with manifest.open("r", encoding="utf-8") as src, output_manifest.open("w", encoding="utf-8") as dst:
        for index, line in enumerate(src):
            if not line.strip():
                continue
            item = json.loads(line)
            uniq_id = chunk_uniq_id(item, index)
            if uniq_id in used_ids:
                uniq_id = f"{uniq_id}_{index:04d}"
            used_ids.add(uniq_id)
            item["uniq_id"] = uniq_id
            item["audio_filepath"] = resolve_manifest_path(item.get("audio_filepath"), manifest)
            item["rttm_filepath"] = resolve_manifest_path(item.get("rttm_filepath"), manifest)
            item["uem_filepath"] = resolve_manifest_path(item.get("uem_filepath"), manifest)
            dst.write(json.dumps(item, ensure_ascii=False) + "\n")

    return output_manifest


def load_model(model_path: Path, SortformerEncLabelModel):
    model_path_str = str(model_path)
    if model_path.suffix == ".nemo":
        return SortformerEncLabelModel.restore_from(model_path_str)
    if model_path.suffix == ".ckpt":
        return SortformerEncLabelModel.load_from_checkpoint(
            checkpoint_path=model_path_str,
            map_location="cpu",
            strict=False,
        )
    return SortformerEncLabelModel.from_pretrained(model_path_str)


def configure_test_data(model, manifest: Path, args: argparse.Namespace, OmegaConf) -> None:
    OmegaConf.set_struct(model._cfg, False)
    if "test_ds" not in model._cfg or model._cfg.test_ds is None:
        model._cfg.test_ds = {}
    model._cfg.test_ds.manifest_filepath = str(manifest.resolve()).replace("\\", "/")
    model._cfg.test_ds.sample_rate = 16000
    model._cfg.test_ds.batch_size = args.batch_size
    model._cfg.test_ds.num_workers = args.num_workers
    model._cfg.test_ds.pin_memory = True
    model._cfg.test_ds.soft_label_thres = 0.5
    model._cfg.test_ds.session_len_sec = -1.0
    model._cfg.test_ds.num_spks = args.max_speakers
    model._cfg.test_ds.soft_targets = False
    model._cfg.test_ds.use_lhotse = False
    model._cfg.test_ds.drop_last = False
    model._cfg.max_num_of_spks = args.max_speakers
    if "sortformer_modules" in model._cfg:
        model._cfg.sortformer_modules.num_spks = args.max_speakers
    OmegaConf.set_struct(model._cfg, True)
    model.setup_test_data(model._cfg.test_ds)


def collect_predictions(model, torch, tqdm) -> list:
    preds_total = []
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(model.test_dataloader(), desc="Sortformer eval"):
            audio_signal, audio_signal_length, _targets, _target_lens = batch
            audio_signal = audio_signal.to(model.device)
            audio_signal_length = audio_signal_length.to(model.device)
            preds = model.forward(audio_signal=audio_signal, audio_signal_length=audio_signal_length)
            preds = preds.detach().cpu()
            if preds.shape[0] == 1:
                preds_total.append(preds)
            else:
                preds_total.extend(torch.split(preds, [1] * preds.shape[0]))
    return preds_total


def print_prediction_stats(preds_total: list, torch) -> None:
    if not preds_total:
        print("Prediction stats: no predictions")
        return

    flat_preds = torch.cat([pred.reshape(-1, pred.shape[-1]).float() for pred in preds_total], dim=0)
    global_stats = {
        "min": flat_preds.min().item(),
        "mean": flat_preds.mean().item(),
        "p95": torch.quantile(flat_preds, 0.95).item(),
        "p99": torch.quantile(flat_preds, 0.99).item(),
        "max": flat_preds.max().item(),
    }
    print(
        "Prediction stats: "
        f"min={global_stats['min']:.4f} mean={global_stats['mean']:.4f} "
        f"p95={global_stats['p95']:.4f} p99={global_stats['p99']:.4f} max={global_stats['max']:.4f}"
    )
    spk_max = flat_preds.max(dim=0).values.tolist()
    print("Speaker max probs: " + " ".join(f"spk{i}={value:.4f}" for i, value in enumerate(spk_max)))


def configure_postprocessing(postprocessing_cfg, args: argparse.Namespace) -> bool:
    if args.onset is not None:
        postprocessing_cfg.onset = args.onset
    if args.offset is not None:
        postprocessing_cfg.offset = args.offset

    bypass_postprocessing = args.bypass_postprocessing
    if args.onset is not None or args.offset is not None or args.postprocessing_yaml is not None:
        bypass_postprocessing = False
    print(
        "Postprocessing: "
        f"bypass={bypass_postprocessing} onset={postprocessing_cfg.onset} offset={postprocessing_cfg.offset}"
    )
    return bypass_postprocessing


def main() -> None:
    args = parse_args()
    deps = import_nemo_deps()
    torch = deps["torch"]

    model_path = resolve_project_path(args.model_path)
    manifest = resolve_project_path(args.manifest)
    out_rttm_dir = resolve_project_path(args.out_rttm_dir)
    out_rttm_dir.mkdir(parents=True, exist_ok=True)

    uniq_manifest = write_manifest_with_unique_ids(manifest)
    audio_map = deps["audio_rttm_map"](str(uniq_manifest))

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 0 else "cpu")
    model = load_model(model_path, deps["SortformerEncLabelModel"])
    configure_test_data(model, uniq_manifest, args, deps["OmegaConf"])
    model.to(device)

    preds_total = collect_predictions(model, torch, deps["tqdm"])
    print_prediction_stats(preds_total, torch)
    postprocessing_cfg = deps["load_postprocessing_from_yaml"](
        None if args.postprocessing_yaml is None else str(resolve_project_path(args.postprocessing_yaml))
    )
    bypass_postprocessing = configure_postprocessing(postprocessing_cfg, args)
    all_hypothesis, all_reference, all_uem = deps["convert_pred_mat_to_segments"](
        audio_rttm_map_dict=audio_map,
        postprocessing_cfg=postprocessing_cfg,
        batch_preds_list=preds_total,
        unit_10ms_frame_count=8,
        bypass_postprocessing=bypass_postprocessing,
        out_rttm_dir=str(out_rttm_dir),
    )
    print(f"Scoring items: hypothesis={len(all_hypothesis)} reference={len(all_reference)} uem={len(all_uem)}")
    score_result = deps["score_labels"](
        AUDIO_RTTM_MAP=audio_map,
        all_reference=all_reference,
        all_hypothesis=all_hypothesis,
        all_uem=all_uem,
        collar=args.collar,
        ignore_overlap=args.ignore_overlap,
        verbose=args.verbose_report,
    )
    if score_result is None:
        raise RuntimeError(
            "DER scoring failed because reference and hypothesis counts differ. "
            "Check whether rttm_filepath values in the generated *_uniq.json exist."
        )
    metric, _mapping, itemized = score_result
    der, cer, fa, miss = itemized
    print(f"| FA: {fa:.4f} | MISS: {miss:.4f} | CER: {cer:.4f} | DER: {der:.4f} |")
    print(f"RTTM output: {out_rttm_dir}")
    print(f"Unique manifest: {uniq_manifest}")


if __name__ == "__main__":
    main()
