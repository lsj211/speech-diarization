"""Sortformer tracklets + Titanet embeddings + NeMo NME-SC clustering.

Pipeline:
1. Run Sortformer on chunked audio and write local chunk RTTMs.
2. Treat each chunk-local speaker stream as a local tracklet.
3. Extract Titanet embeddings from clean subsegments inside each tracklet.
4. Use NeMo's LongFormSpeakerClustering for NME-SC speaker counting/clustering.
5. Vote subsegment cluster labels back to tracklets and re-label Sortformer RTTM boundaries.

Compared with sortformer_tracklet_cluster.py, this keeps the Sortformer/Titanet
front end but delegates the main speaker clustering step to NeMo baseline code.
"""

from __future__ import annotations

import argparse
import importlib
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# Avoid import-time torch.compile/Triton failures in some NeMo/PyTorch builds.
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

from eval_sortformer_6spk import (
    audio_duration,
    build_single_session_chunk_manifest,
    find_source_item,
    resolve_session_file,
)
from eval_sortformer_8spk import (
    NEMO_ROOT,
    collect_predictions,
    configure_postprocessing,
    configure_test_data,
    import_nemo_deps,
    load_model,
    print_prediction_stats,
    resolve_project_path,
    write_manifest_with_unique_ids,
)
from sortformer_tracklet_cluster import (
    chunk_metadata,
    disable_torch_compile,
    force_cpu_cuda_state,
    import_speaker_model,
    label_speaker_count,
    load_speaker_model,
    read_jsonl,
    read_rttm_segments,
    reference_speaker_count,
    write_embedding_manifest,
    write_global_rttm,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sortformer tracklets + Titanet + NeMo NME-SC clustering.")
    parser.add_argument("--sortformer-model-path", type=Path, default=Path("models/diar_sortformer_4spk-v1.nemo"))
    parser.add_argument("--speaker-model-path", default="titanet_large")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional prebuilt Sortformer chunk manifest.")
    parser.add_argument("--source-manifest", type=Path, default=Path("data/manifests/aishell4_test_manifest_mono.json"))
    parser.add_argument("--session-id", default="L_R003S01C02")
    parser.add_argument("--chunk-sec", type=float, default=120.0)
    parser.add_argument("--chunk-hop-sec", type=float, default=120.0)
    parser.add_argument(
        "--generated-manifest",
        type=Path,
        default=Path("data/manifests/model_training/aishell4_test_nemo_nmesc_chunks.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/model_training/sortformer_nemo_nmesc_cluster/L_R003S01C02"),
    )
    parser.add_argument("--max-speakers", type=int, default=4, help="Sortformer output speaker capacity.")
    parser.add_argument("--max-global-speakers", type=int, default=8, help="NME-SC upper bound when not using oracle.")
    parser.add_argument("--oracle-num-speakers", action="store_true", help="Use reference RTTM speaker count.")
    parser.add_argument("--max-rp-threshold", type=float, default=0.25)
    parser.add_argument("--sparse-search-volume", type=int, default=30)
    parser.add_argument("--enhanced-count-thres", type=int, default=80)
    parser.add_argument("--fixed-thres", type=float, default=-1.0)
    parser.add_argument("--kmeans-random-trials", type=int, default=1)
    parser.add_argument("--chunk-cluster-count", type=int, default=50)
    parser.add_argument("--embeddings-per-chunk", type=int, default=10000)
    parser.add_argument("--embedding-window", type=float, default=1.5)
    parser.add_argument("--embedding-shift", type=float, default=0.75)
    parser.add_argument("--min-embedding-duration", type=float, default=0.5)
    parser.add_argument("--drop-overlap-embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--cluster-on-cpu", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--collar", type=float, default=0.25)
    parser.add_argument("--onset", type=float, default=None)
    parser.add_argument("--offset", type=float, default=None)
    parser.add_argument("--ignore-overlap", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bypass-postprocessing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--postprocessing-yaml", type=Path, default=None)
    parser.add_argument("--verbose-report", action="store_true")
    return parser.parse_args()


def extract_ordered_embeddings(manifest: Path, speaker_model, torch, args: argparse.Namespace) -> tuple[list[dict], np.ndarray]:
    speaker_model.setup_test_data(
        {
            "manifest_filepath": str(manifest.resolve()).replace("\\", "/"),
            "sample_rate": 16000,
            "batch_size": args.embedding_batch_size,
            "trim_silence": False,
            "labels": None,
            "num_workers": args.num_workers,
        }
    )

    rows = read_jsonl(manifest)
    all_embeddings = []
    with torch.inference_mode():
        for batch in speaker_model.test_dataloader():
            batch = [value.to(speaker_model.device) for value in batch]
            audio_signal, audio_signal_len, _labels, _slices = batch
            _logits, embeddings = speaker_model.forward(
                input_signal=audio_signal,
                input_signal_length=audio_signal_len,
            )
            all_embeddings.extend(list(embeddings.detach().float().cpu().numpy()))

    if len(all_embeddings) != len(rows):
        raise RuntimeError(f"Embedding count mismatch: got {len(all_embeddings)}, manifest rows {len(rows)}")
    return rows, np.stack(all_embeddings, axis=0)


def run_nemo_nmesc(rows: list[dict], embeddings: np.ndarray, torch, args: argparse.Namespace, oracle_count: int | None):
    module = importlib.import_module("nemo.collections.asr.parts.utils.longform_clustering")
    module_path = Path(module.__file__).resolve()
    nemo_root = NEMO_ROOT.resolve()
    if not str(module_path).startswith(str(nemo_root)):
        raise RuntimeError(f"Imported NeMo clustering from {module_path}, expected local source under {nemo_root}")
    LongFormSpeakerClustering = module.LongFormSpeakerClustering

    timestamps = []
    for row in rows:
        start = float(row["offset"])
        end = start + float(row["duration"])
        timestamps.append([start, end])

    use_cuda = torch.cuda.is_available() and args.cuda >= 0 and not args.cluster_on_cpu
    clustering = LongFormSpeakerClustering(cuda=use_cuda)
    cluster_labels = clustering.forward_infer(
        embeddings_in_scales=torch.tensor(embeddings, dtype=torch.float32),
        timestamps_in_scales=torch.tensor(timestamps, dtype=torch.float32),
        multiscale_segment_counts=torch.LongTensor([len(rows)]),
        multiscale_weights=torch.tensor([[1.0]], dtype=torch.float32),
        oracle_num_speakers=-1 if oracle_count is None else int(oracle_count),
        max_num_speakers=int(args.max_global_speakers),
        max_rp_threshold=float(args.max_rp_threshold),
        enhanced_count_thres=int(args.enhanced_count_thres),
        sparse_search_volume=int(args.sparse_search_volume),
        fixed_thres=float(args.fixed_thres),
        chunk_cluster_count=int(args.chunk_cluster_count),
        embeddings_per_chunk=int(args.embeddings_per_chunk),
    )
    return cluster_labels.detach().cpu().numpy().astype(int).tolist()


def vote_labels_to_tracklets(rows: list[dict], subsegment_labels: list[int]) -> dict[str, int]:
    count_votes: dict[str, Counter] = defaultdict(Counter)
    duration_votes: dict[str, Counter] = defaultdict(Counter)
    for row, label in zip(rows, subsegment_labels):
        tracklet_id = str(row["uniq_id"])
        duration = float(row["duration"])
        count_votes[tracklet_id][int(label)] += 1
        duration_votes[tracklet_id][int(label)] += duration

    labels_by_tracklet = {}
    for tracklet_id in sorted(count_votes):
        best_label = max(
            count_votes[tracklet_id],
            key=lambda label: (count_votes[tracklet_id][label], duration_votes[tracklet_id][label], -label),
        )
        labels_by_tracklet[tracklet_id] = int(best_label)
    return labels_by_tracklet


def print_clustering_summary(rows: list[dict], subsegment_labels: list[int], labels_by_tracklet: dict[str, int]) -> None:
    print(
        "NeMo NME-SC clustering: "
        f"embedding_rows={len(rows)} embedded_tracklets={len(labels_by_tracklet)} "
        f"subsegment_speakers={len(set(subsegment_labels))} "
        f"tracklet_speakers={len(set(labels_by_tracklet.values()))}"
    )


def main() -> None:
    args = parse_args()
    if args.cuda < 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import torch as torch_module

    disable_torch_compile(torch_module)
    if args.cuda < 0:
        force_cpu_cuda_state(torch_module)
    deps = import_nemo_deps()
    torch = deps["torch"]
    disable_torch_compile(torch)
    if args.cuda < 0:
        force_cpu_cuda_state(torch)
    from nemo.collections.asr.parts.utils.speaker_utils import labels_to_supervisions, rttm_to_labels

    sortformer_model_path = resolve_project_path(args.sortformer_model_path)
    output_dir = resolve_project_path(args.output_dir)
    sortformer_rttm_dir = output_dir / "sortformer_chunk_rttm"
    global_rttm_dir = output_dir / "global_rttm"
    output_dir.mkdir(parents=True, exist_ok=True)
    sortformer_rttm_dir.mkdir(parents=True, exist_ok=True)
    global_rttm_dir.mkdir(parents=True, exist_ok=True)

    if args.manifest is None:
        manifest = build_single_session_chunk_manifest(args)
    else:
        manifest = resolve_project_path(args.manifest)
    uniq_manifest = write_manifest_with_unique_ids(manifest)
    chunk_meta = chunk_metadata(uniq_manifest)
    audio_map = deps["audio_rttm_map"](str(uniq_manifest))

    source_manifest = resolve_project_path(args.source_manifest)
    source_item = find_source_item(source_manifest, args.session_id)
    audio_path = resolve_session_file(source_item, "audio_filepath", args.session_id, "wavs_mono", ".flac")
    rttm_path = resolve_session_file(source_item, "rttm_filepath", args.session_id, "rttm", ".rttm")
    session_duration = audio_duration(audio_path, rttm_path)

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 0 else "cpu")
    sortformer = load_model(sortformer_model_path, deps["SortformerEncLabelModel"])
    configure_test_data(sortformer, uniq_manifest, args, deps["OmegaConf"])
    sortformer.to(device)

    preds_total = collect_predictions(sortformer, torch, deps["tqdm"])
    print_prediction_stats(preds_total, torch)
    postprocessing_cfg = deps["load_postprocessing_from_yaml"](
        None if args.postprocessing_yaml is None else str(resolve_project_path(args.postprocessing_yaml))
    )
    bypass_postprocessing = configure_postprocessing(postprocessing_cfg, args)
    deps["convert_pred_mat_to_segments"](
        audio_rttm_map_dict=audio_map,
        postprocessing_cfg=postprocessing_cfg,
        batch_preds_list=preds_total,
        unit_10ms_frame_count=8,
        bypass_postprocessing=bypass_postprocessing,
        out_rttm_dir=str(sortformer_rttm_dir),
    )

    segments = read_rttm_segments(sortformer_rttm_dir, chunk_meta, args.session_id)
    if not segments:
        raise RuntimeError(f"No Sortformer segments found in {sortformer_rttm_dir}")
    print(f"Sortformer local segments: {len(segments)}")

    embedding_manifest = output_dir / "tracklet_embedding_manifest.json"
    tracklet_counts = write_embedding_manifest(segments, audio_path, embedding_manifest, args)
    if not tracklet_counts:
        raise RuntimeError("No valid embedding subsegments were generated. Lower --min-embedding-duration.")
    print(f"Embedding manifest: {embedding_manifest} ({sum(tracklet_counts.values())} rows)")

    speaker_model_cls = import_speaker_model()
    speaker_model = load_speaker_model(args.speaker_model_path, speaker_model_cls, device)
    rows, embeddings = extract_ordered_embeddings(embedding_manifest, speaker_model, torch, args)

    oracle_count = reference_speaker_count(rttm_path) if args.oracle_num_speakers else None
    subsegment_labels = run_nemo_nmesc(rows, embeddings, torch, args, oracle_count)
    labels_by_tracklet = vote_labels_to_tracklets(rows, subsegment_labels)
    print_clustering_summary(rows, subsegment_labels, labels_by_tracklet)
    if oracle_count is not None:
        print(f"Oracle speaker count: {oracle_count}")

    global_rttm_path = global_rttm_dir / f"{args.session_id}.rttm"
    hyp_labels = write_global_rttm(segments, labels_by_tracklet, global_rttm_path)
    print(f"Global RTTM: {global_rttm_path}")

    session_audio_map = {
        args.session_id: {
            "audio_filepath": str(audio_path.resolve()).replace("\\", "/"),
            "rttm_filepath": str(rttm_path.resolve()).replace("\\", "/"),
            "offset": 0.0,
            "duration": session_duration,
            "num_speakers": reference_speaker_count(rttm_path),
            "uem_filepath": None,
        }
    }
    ref_labels = rttm_to_labels(str(rttm_path))
    score_result = deps["score_labels"](
        AUDIO_RTTM_MAP=session_audio_map,
        all_reference=[[args.session_id, labels_to_supervisions(ref_labels, uniq_name=args.session_id)]],
        all_hypothesis=[[args.session_id, labels_to_supervisions(hyp_labels, uniq_name=args.session_id)]],
        all_uem=None,
        collar=args.collar,
        ignore_overlap=args.ignore_overlap,
        verbose=True,
    )
    if score_result is None:
        raise RuntimeError("DER scoring failed.")
    _metric, _mapping, itemized = score_result
    der, cer, fa, miss = itemized
    ref_count = reference_speaker_count(rttm_path)
    hyp_count = label_speaker_count(hyp_labels)
    print(f"Session speaker count: {args.session_id} ref={ref_count} hyp={hyp_count}")
    print(f"score_labels itemized: FA={fa:.4f} MISS={miss:.4f} CER={cer:.4f} DER={der:.4f}")


if __name__ == "__main__":
    main()
