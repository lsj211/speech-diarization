"""Evaluate a 6-speaker Sortformer checkpoint on one AISHELL-4 session.

This wrapper keeps the 8spk evaluation mechanics but sets safer defaults for
the 6spk original-ATS/PIL experiment and builds a chunked manifest for the
baseline session L_R003S01C02 by default.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from eval_sortformer_8spk import (
    PROJECT_ROOT,
    collect_predictions,
    configure_postprocessing,
    configure_test_data,
    import_nemo_deps,
    load_model,
    print_prediction_stats,
    resolve_project_path,
    write_manifest_with_unique_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned 6-speaker Sortformer.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(
            "results/model_training/sortformer_6spk_original_smoke/"
            "sortformer_6spk_original_aishell4_train_LM.nemo"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional prebuilt chunked manifest. If omitted, build one from --source-manifest.",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("data/manifests/aishell4_test_manifest_mono.json"),
        help="Full-session AISHELL-4 test manifest used to locate --session-id.",
    )
    parser.add_argument("--session-id", default="L_R003S01C02")
    parser.add_argument("--chunk-sec", type=float, default=45.0)
    parser.add_argument("--chunk-hop-sec", type=float, default=45.0)
    parser.add_argument(
        "--generated-manifest",
        type=Path,
        default=Path("data/manifests/model_training/aishell4_test_L_R003S01C02_6spk_c45_chunks.json"),
    )
    parser.add_argument(
        "--out-rttm-dir",
        type=Path,
        default=Path("results/model_training/sortformer_6spk_original_smoke/eval_L_R003S01C02"),
    )
    parser.add_argument("--max-speakers", type=int, default=6)
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
    parser.add_argument(
        "--speaker-count-report-limit",
        type=int,
        default=40,
        help="Maximum number of chunk-level speaker-count rows to print. Use 0 to disable chunk rows.",
    )
    return parser.parse_args()


def session_stem(value: str | None) -> str:
    if not value:
        return ""
    return Path(value).stem


def find_source_item(source_manifest: Path, session_id: str) -> dict:
    with source_manifest.open("r", encoding="utf-8") as src:
        for line in src:
            if not line.strip():
                continue
            item = json.loads(line)
            stems = {
                str(item.get("uniq_id") or ""),
                session_stem(item.get("audio_filepath")),
                session_stem(item.get("rttm_filepath")),
            }
            if session_id in stems:
                return item
    raise FileNotFoundError(f"Session {session_id} not found in {source_manifest}")


def project_session_path(subdir: str, session_id: str, suffix: str) -> Path:
    return PROJECT_ROOT / "data" / subdir / f"{session_id}{suffix}"


def resolve_session_file(item: dict, key: str, session_id: str, subdir: str, suffix: str) -> Path:
    project_path = project_session_path(subdir, session_id, suffix)
    if project_path.exists():
        return project_path

    value = item.get(key)
    if value:
        path = Path(value)
        if path.exists():
            return path

    return project_path


def rttm_duration(rttm_path: Path) -> float:
    max_end = 0.0
    with rttm_path.open("r", encoding="utf-8") as rttm:
        for line in rttm:
            parts = line.strip().split()
            if len(parts) < 5 or parts[0] != "SPEAKER":
                continue
            start = float(parts[3])
            duration = float(parts[4])
            max_end = max(max_end, start + duration)
    if max_end <= 0.0:
        raise ValueError(f"No RTTM speech segments found in {rttm_path}")
    return max_end


def audio_duration(audio_path: Path, rttm_path: Path) -> float:
    try:
        import soundfile as sf

        return float(sf.info(str(audio_path)).duration)
    except Exception:
        return rttm_duration(rttm_path)


def build_single_session_chunk_manifest(args: argparse.Namespace) -> Path:
    source_manifest = resolve_project_path(args.source_manifest)
    generated_manifest = resolve_project_path(args.generated_manifest)
    generated_manifest.parent.mkdir(parents=True, exist_ok=True)

    item = find_source_item(source_manifest, args.session_id)
    audio_path = resolve_session_file(item, "audio_filepath", args.session_id, "wavs_mono", ".flac")
    rttm_path = resolve_session_file(item, "rttm_filepath", args.session_id, "rttm", ".rttm")
    duration = audio_duration(audio_path, rttm_path)

    offset = 0.0
    index = 0
    with generated_manifest.open("w", encoding="utf-8") as dst:
        while offset < duration:
            chunk_duration = min(args.chunk_sec, duration - offset)
            if chunk_duration <= 0.0:
                break
            chunk = {
                "audio_filepath": str(audio_path.resolve()).replace("\\", "/"),
                "offset": round(offset, 4),
                "duration": round(chunk_duration, 4),
                "label": "infer",
                "text": "-",
                "num_speakers": args.max_speakers,
                "rttm_filepath": str(rttm_path.resolve()).replace("\\", "/"),
                "uem_filepath": None,
                "uniq_id": f"{args.session_id}_chunk{index:04d}",
            }
            dst.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            offset += args.chunk_hop_sec
            index += 1

    print(f"Generated chunk manifest: {generated_manifest} ({index} chunks)")
    return generated_manifest


def session_id_from_chunk_id(uniq_id: str) -> str:
    return re.sub(r"_chunk\d{4}.*$", "", uniq_id)


def speaker_count_rows(all_reference: list, all_hypothesis: list, unique_speakers) -> list[dict]:
    rows = []
    hyp_by_id = {str(uniq_id): labels for uniq_id, labels in all_hypothesis}
    for uniq_id, ref_labels in all_reference:
        uniq_id = str(uniq_id)
        hyp_labels = hyp_by_id.get(uniq_id, [])
        ref_speakers = sorted(str(speaker) for speaker in unique_speakers(ref_labels))
        hyp_speakers = sorted(str(speaker) for speaker in unique_speakers(hyp_labels))
        rows.append(
            {
                "uniq_id": uniq_id,
                "session_id": session_id_from_chunk_id(uniq_id),
                "ref_speakers": ref_speakers,
                "hyp_speakers": hyp_speakers,
            }
        )
    return rows


def print_speaker_count_diagnostics(
    all_reference: list,
    all_hypothesis: list,
    unique_speakers,
    report_limit: int,
) -> None:
    rows = speaker_count_rows(all_reference, all_hypothesis, unique_speakers)
    if not rows:
        print("Speaker count diagnostics: no reference/hypothesis rows")
        return

    matched = sum(len(row["ref_speakers"]) == len(row["hyp_speakers"]) for row in rows)
    print(
        "Speaker count diagnostics: "
        f"chunk_exact_match={matched}/{len(rows)} acc={matched / len(rows):.4f}"
    )

    session_speakers: dict[str, dict[str, set[str]]] = {}
    for row in rows:
        session = session_speakers.setdefault(row["session_id"], {"ref": set(), "hyp": set()})
        session["ref"].update(row["ref_speakers"])
        session["hyp"].update(row["hyp_speakers"])

    for session_id, speakers in sorted(session_speakers.items()):
        ref_speakers = sorted(speakers["ref"])
        hyp_speakers = sorted(speakers["hyp"])
        print(
            f"Session speaker count: {session_id} ref={len(ref_speakers)} hyp={len(hyp_speakers)} "
            f"ref_spks={','.join(ref_speakers) or '-'} hyp_spks={','.join(hyp_speakers) or '-'}"
        )

    if report_limit <= 0:
        return

    print(f"Chunk speaker counts (first {min(report_limit, len(rows))}/{len(rows)}):")
    for row in rows[:report_limit]:
        ref_speakers = row["ref_speakers"]
        hyp_speakers = row["hyp_speakers"]
        print(
            f"  {row['uniq_id']}: ref={len(ref_speakers)} hyp={len(hyp_speakers)} "
            f"ref_spks={','.join(ref_speakers) or '-'} hyp_spks={','.join(hyp_speakers) or '-'}"
        )


def main() -> None:
    args = parse_args()
    deps = import_nemo_deps()
    torch = deps["torch"]
    from nemo.collections.asr.metrics.der import unique_speakers

    model_path = resolve_project_path(args.model_path)
    manifest = resolve_project_path(args.manifest) if args.manifest is not None else build_single_session_chunk_manifest(args)
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
    print_speaker_count_diagnostics(
        all_reference=all_reference,
        all_hypothesis=all_hypothesis,
        unique_speakers=unique_speakers,
        report_limit=args.speaker_count_report_limit,
    )
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
    _metric, _mapping, itemized = score_result
    der, cer, fa, miss = itemized
    print(f"| FA: {fa:.4f} | MISS: {miss:.4f} | CER: {cer:.4f} | DER: {der:.4f} |")
    print(f"RTTM output: {out_rttm_dir}")
    print(f"Unique manifest: {uniq_manifest}")


if __name__ == "__main__":
    main()
