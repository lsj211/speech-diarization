"""Link chunk-level Sortformer tracklets into session-level speakers.

Pipeline:
1. Run Sortformer on chunked audio and write local chunk RTTMs.
2. Treat each chunk-local speaker stream as a local tracklet.
3. Extract Titanet embeddings from clean subsegments inside each tracklet.
4. Cluster tracklet embeddings into global speakers.
5. Re-label the original Sortformer RTTM boundaries with global speaker ids.

This keeps Sortformer's local timing/overlap decisions while making speaker
labels meaningful across chunks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
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


@dataclass
class Segment:
    chunk_id: str
    session_id: str
    local_speaker: str
    start: float
    end: float
    tracklet_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sortformer local diarization + Titanet tracklet clustering.")
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
        default=Path("data/manifests/model_training/aishell4_test_tracklet_cluster_chunks.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/model_training/sortformer_tracklet_cluster/L_R003S01C02"),
    )
    parser.add_argument("--max-speakers", type=int, default=4, help="Sortformer output speaker capacity.")
    parser.add_argument("--max-global-speakers", type=int, default=8, help="Upper bound for auto clustering.")
    parser.add_argument("--oracle-num-speakers", action="store_true", help="Use reference RTTM speaker count.")
    parser.add_argument("--clustering-threshold", type=float, default=0.35, help="Cosine distance merge threshold.")
    parser.add_argument("--embedding-window", type=float, default=1.5)
    parser.add_argument("--embedding-shift", type=float, default=0.75)
    parser.add_argument("--min-embedding-duration", type=float, default=0.5)
    parser.add_argument("--drop-overlap-embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--collar", type=float, default=0.25)
    parser.add_argument("--onset", type=float, default=None)
    parser.add_argument("--offset", type=float, default=None)
    parser.add_argument("--ignore-overlap", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bypass-postprocessing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--postprocessing-yaml", type=Path, default=None)
    parser.add_argument("--verbose-report", action="store_true")
    return parser.parse_args()


def import_speaker_model():
    nemo_root = str(NEMO_ROOT)
    if nemo_root not in sys.path:
        sys.path.insert(0, nemo_root)

    from nemo.collections.asr.models import EncDecSpeakerLabelModel

    return EncDecSpeakerLabelModel


def disable_torch_compile(torch_module) -> None:
    """Make torch.compile a no-op before importing NeMo modules that call it at import time."""
    if getattr(torch_module, "_sortformer_tracklet_compile_disabled", False):
        return

    def _no_compile(model=None, *args, **kwargs):
        if model is None:
            return lambda fn: fn
        return model

    torch_module.compile = _no_compile
    torch_module._sortformer_tracklet_compile_disabled = True


def force_cpu_cuda_state(torch_module) -> None:
    """Hide broken CUDA runtimes from NeMo when the user requests CPU mode."""
    torch_module.cuda.is_available = lambda: False
    torch_module.cuda.current_device = lambda: None


def load_speaker_model(model_path: str, model_cls, device):
    if model_path.endswith(".nemo"):
        model = model_cls.restore_from(model_path, map_location=device)
    elif model_path.endswith(".ckpt"):
        model = model_cls.load_from_checkpoint(model_path, map_location=device)
    else:
        try:
            model = model_cls.from_pretrained(model_name=model_path, map_location=device)
        except TypeError:
            model = model_cls.from_pretrained(model_path)
    return model.to(device).eval()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as src:
        for line in src:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def chunk_metadata(manifest: Path) -> dict[str, dict]:
    rows = read_jsonl(manifest)
    metadata = {}
    for index, item in enumerate(rows):
        uniq_id = item.get("uniq_id")
        if uniq_id is None:
            uniq_id = f"{Path(item['audio_filepath']).stem}_chunk{index:04d}"
        metadata[str(uniq_id)] = item
    return metadata


def read_rttm_segments(rttm_dir: Path, metadata: dict[str, dict], session_id: str) -> list[Segment]:
    segments = []
    for chunk_id, item in metadata.items():
        rttm_path = rttm_dir / f"{chunk_id}.rttm"
        if not rttm_path.exists():
            continue
        with rttm_path.open("r", encoding="utf-8") as src:
            for line in src:
                parts = line.strip().split()
                if len(parts) < 8 or parts[0] != "SPEAKER":
                    continue
                # NeMo's predlist_to_timestamps already adds the manifest chunk offset.
                start = float(parts[3])
                duration = float(parts[4])
                local_speaker = parts[7]
                end = start + duration
                tracklet_id = f"{chunk_id}__{local_speaker}"
                segments.append(
                    Segment(
                        chunk_id=chunk_id,
                        session_id=session_id,
                        local_speaker=local_speaker,
                        start=round(start, 3),
                        end=round(end, 3),
                        tracklet_id=tracklet_id,
                    )
                )
    return sorted(segments, key=lambda seg: (seg.start, seg.end, seg.local_speaker))


def overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)


def is_clean_subsegment(candidate: tuple[float, float], segment: Segment, all_segments: list[Segment]) -> bool:
    cand_start, cand_end = candidate
    for other in all_segments:
        if other.chunk_id != segment.chunk_id or other.local_speaker == segment.local_speaker:
            continue
        if overlaps(cand_start, cand_end, other.start, other.end):
            return False
    return True


def subsegments(start: float, end: float, window: float, shift: float, min_duration: float) -> list[tuple[float, float]]:
    duration = end - start
    if duration < min_duration:
        return []
    if duration <= window:
        return [(start, end)]

    output = []
    cursor = start
    while cursor + min_duration <= end:
        seg_end = min(cursor + window, end)
        if seg_end - cursor >= min_duration:
            output.append((round(cursor, 3), round(seg_end, 3)))
        cursor += shift
    return output


def write_embedding_manifest(
    segments: list[Segment],
    audio_path: Path,
    output_path: Path,
    args: argparse.Namespace,
) -> dict[str, int]:
    tracklet_counts: dict[str, int] = {}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as dst:
        for segment in segments:
            candidates = subsegments(
                start=segment.start,
                end=segment.end,
                window=args.embedding_window,
                shift=args.embedding_shift,
                min_duration=args.min_embedding_duration,
            )
            if args.drop_overlap_embeddings:
                candidates = [cand for cand in candidates if is_clean_subsegment(cand, segment, segments)]
            for emb_start, emb_end in candidates:
                item = {
                    "audio_filepath": str(audio_path.resolve()).replace("\\", "/"),
                    "offset": round(emb_start, 3),
                    "duration": round(emb_end - emb_start, 3),
                    "label": segment.tracklet_id,
                    "text": "-",
                    "uniq_id": segment.tracklet_id,
                }
                dst.write(json.dumps(item, ensure_ascii=False) + "\n")
                tracklet_counts[segment.tracklet_id] = tracklet_counts.get(segment.tracklet_id, 0) + 1
    return tracklet_counts


def extract_embeddings(manifest: Path, speaker_model, torch, args: argparse.Namespace) -> dict[str, list[np.ndarray]]:
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
            embeddings = embeddings.detach().float().cpu().numpy()
            all_embeddings.extend(list(embeddings))

    if len(all_embeddings) != len(rows):
        raise RuntimeError(f"Embedding count mismatch: got {len(all_embeddings)}, manifest rows {len(rows)}")

    by_tracklet: dict[str, list[np.ndarray]] = {}
    for row, embedding in zip(rows, all_embeddings):
        by_tracklet.setdefault(str(row["uniq_id"]), []).append(embedding)
    return by_tracklet


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm <= 0:
        return vector
    return vector / norm


def average_tracklet_embeddings(by_tracklet: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    averaged = {}
    for tracklet_id, embeddings in by_tracklet.items():
        stacked = np.stack([normalize(emb) for emb in embeddings], axis=0)
        averaged[tracklet_id] = normalize(stacked.mean(axis=0))
    return averaged


def cluster_distance(cluster_a: list[int], cluster_b: list[int], distance_matrix: np.ndarray) -> float:
    values = [distance_matrix[i, j] for i in cluster_a for j in cluster_b]
    return float(np.mean(values))


def build_cannot_link_pairs(segments: list[Segment], tracklet_ids: list[str]) -> set[tuple[int, int]]:
    """Prevent merging different Sortformer streams from the same chunk."""
    tracklet_index = {tracklet_id: index for index, tracklet_id in enumerate(tracklet_ids)}
    by_chunk: dict[str, set[str]] = {}
    for segment in segments:
        by_chunk.setdefault(segment.chunk_id, set()).add(segment.tracklet_id)

    cannot_link = set()
    for chunk_tracklets in by_chunk.values():
        ordered = sorted(tracklet for tracklet in chunk_tracklets if tracklet in tracklet_index)
        for left_pos, left in enumerate(ordered):
            for right in ordered[left_pos + 1 :]:
                pair = tuple(sorted((tracklet_index[left], tracklet_index[right])))
                cannot_link.add(pair)
    return cannot_link


def clusters_can_merge(cluster_a: list[int], cluster_b: list[int], cannot_link: set[tuple[int, int]]) -> bool:
    for left in cluster_a:
        for right in cluster_b:
            if tuple(sorted((left, right))) in cannot_link:
                return False
    return True


def agglomerative_labels(
    embeddings: np.ndarray,
    max_speakers: int,
    threshold: float,
    oracle_num_speakers: int | None,
    cannot_link: set[tuple[int, int]] | None = None,
) -> list[int]:
    cannot_link = cannot_link or set()
    n_items = embeddings.shape[0]
    if n_items == 0:
        return []
    if n_items == 1:
        return [0]

    similarity = np.clip(embeddings @ embeddings.T, -1.0, 1.0)
    distance = 1.0 - similarity
    clusters = [[idx] for idx in range(n_items)]
    target = oracle_num_speakers

    while len(clusters) > 1:
        best_pair = None
        best_distance = math.inf
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if not clusters_can_merge(clusters[i], clusters[j], cannot_link):
                    continue
                dist = cluster_distance(clusters[i], clusters[j], distance)
                if dist < best_distance:
                    best_distance = dist
                    best_pair = (i, j)

        should_merge = False
        if target is not None:
            should_merge = len(clusters) > target
        else:
            should_merge = best_distance <= threshold or len(clusters) > max_speakers
        if best_pair is None:
            print(
                "Clustering stopped: cannot-link constraints prevent further merges "
                f"at {len(clusters)} clusters"
            )
            break
        if not should_merge:
            break

        left, right = best_pair
        clusters[left] = clusters[left] + clusters[right]
        del clusters[right]

    labels = [-1] * n_items
    for label, cluster in enumerate(sorted(clusters, key=lambda items: min(items))):
        for item_idx in cluster:
            labels[item_idx] = label
    return labels


def reference_speaker_count(rttm_path: Path) -> int:
    speakers = set()
    with rttm_path.open("r", encoding="utf-8") as src:
        for line in src:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == "SPEAKER":
                speakers.add(parts[7])
    return len(speakers)


def label_speaker_count(labels: list[str]) -> int:
    speakers = set()
    for label in labels:
        parts = label.strip().split()
        if len(parts) >= 3:
            speakers.add(parts[2])
    return len(speakers)


def write_global_rttm(segments: list[Segment], labels_by_tracklet: dict[str, int], output_path: Path) -> list[str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    label_rows = []
    with output_path.open("w", encoding="utf-8") as dst:
        for segment in segments:
            if segment.tracklet_id not in labels_by_tracklet:
                continue
            speaker = f"speaker_{labels_by_tracklet[segment.tracklet_id]}"
            duration = segment.end - segment.start
            if duration <= 0:
                continue
            dst.write(
                f"SPEAKER {segment.session_id} 1   {segment.start:.3f}   {duration:.3f} "
                f"<NA> <NA> {speaker} <NA> <NA>\n"
            )
            label_rows.append(f"{segment.start:.3f} {segment.end:.3f} {speaker}")
    return label_rows


def print_tracklet_summary(segments: list[Segment], labels_by_tracklet: dict[str, int]) -> None:
    local_tracklets = sorted({segment.tracklet_id for segment in segments})
    used_labels = sorted(set(labels_by_tracklet.values()))
    print(
        "Tracklet clustering: "
        f"local_tracklets={len(local_tracklets)} embedded={len(labels_by_tracklet)} "
        f"global_speakers={len(used_labels)}"
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
    by_tracklet = extract_embeddings(embedding_manifest, speaker_model, torch, args)
    averaged = average_tracklet_embeddings(by_tracklet)
    tracklet_ids = sorted(averaged)
    embedding_matrix = np.stack([averaged[tracklet_id] for tracklet_id in tracklet_ids], axis=0)
    cannot_link = build_cannot_link_pairs(segments, tracklet_ids)

    oracle_count = reference_speaker_count(rttm_path) if args.oracle_num_speakers else None
    cluster_labels = agglomerative_labels(
        embedding_matrix,
        max_speakers=args.max_global_speakers,
        threshold=args.clustering_threshold,
        oracle_num_speakers=oracle_count,
        cannot_link=cannot_link,
    )
    labels_by_tracklet = dict(zip(tracklet_ids, cluster_labels))
    print_tracklet_summary(segments, labels_by_tracklet)
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
