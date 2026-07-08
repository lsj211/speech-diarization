"""Link overlapping chunk-level Sortformer speakers without an external embedding model.

Pipeline:
1. Run Sortformer on overlapping chunks and write local chunk RTTMs.
2. Treat each chunk-local speaker stream as a local tracklet.
3. Match local speakers between overlapping chunks using activity overlap.
4. Merge matched tracklets into global speakers with cannot-link constraints.
5. Re-label and union the Sortformer RTTM boundaries into a session-level RTTM.

This keeps the system pure Sortformer: no Titanet or other speaker encoder is
used for global speaker linking.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

# Avoid import-time torch.compile/Triton failures in some NeMo/PyTorch builds.
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

from eval_sortformer_6spk import (
    audio_duration,
    build_single_session_chunk_manifest,
    find_source_item,
    resolve_session_file,
)
from eval_sortformer_8spk import (
    collect_predictions,
    configure_postprocessing,
    configure_test_data,
    import_nemo_deps,
    load_model,
    print_prediction_stats,
    resolve_project_path,
    write_manifest_with_unique_ids,
)


@dataclass(frozen=True)
class ChunkInfo:
    chunk_id: str
    offset: float
    duration: float

    @property
    def end(self) -> float:
        return self.offset + self.duration


@dataclass(frozen=True)
class Segment:
    chunk_id: str
    session_id: str
    local_speaker: str
    start: float
    end: float
    tracklet_id: str


class UnionFind:
    def __init__(self, items: list[str], chunk_by_tracklet: dict[str, str]) -> None:
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}
        self.chunk_members = {item: {chunk_by_tracklet[item]} for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def can_union(self, left: str, right: str) -> bool:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return True
        return self.chunk_members[left_root].isdisjoint(self.chunk_members[right_root])

    def union(self, left: str, right: str) -> bool:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        if not self.can_union(left_root, right_root):
            return False
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.chunk_members[left_root].update(self.chunk_members[right_root])
        del self.chunk_members[right_root]
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1
        return True

    def components(self) -> dict[str, list[str]]:
        output: dict[str, list[str]] = {}
        for item in self.parent:
            output.setdefault(self.find(item), []).append(item)
        return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pure Sortformer overlap speaker linking.")
    parser.add_argument("--sortformer-model-path", type=Path, default=Path("models/diar_sortformer_4spk-v1.nemo"))
    parser.add_argument("--manifest", type=Path, default=None, help="Optional prebuilt overlapping chunk manifest.")
    parser.add_argument("--source-manifest", type=Path, default=Path("data/manifests/aishell4_test_manifest_mono.json"))
    parser.add_argument("--session-id", default="L_R003S01C02")
    parser.add_argument("--chunk-sec", type=float, default=120.0)
    parser.add_argument("--chunk-hop-sec", type=float, default=60.0)
    parser.add_argument(
        "--generated-manifest",
        type=Path,
        default=Path("data/manifests/model_training/aishell4_test_overlap_link_chunks.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/model_training/sortformer_overlap_link/L_R003S01C02"),
    )
    parser.add_argument("--max-speakers", type=int, default=4, help="Sortformer output speaker capacity.")
    parser.add_argument("--link-threshold", type=float, default=0.20, help="Minimum overlap score for linking.")
    parser.add_argument("--min-link-intersection", type=float, default=0.5, help="Minimum shared speech seconds.")
    parser.add_argument("--min-overlap-sec", type=float, default=5.0, help="Minimum chunk overlap to attempt matching.")
    parser.add_argument("--merge-gap", type=float, default=0.05, help="Merge same-speaker intervals separated by this gap.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--collar", type=float, default=0.25)
    parser.add_argument("--onset", type=float, default=None)
    parser.add_argument("--offset", type=float, default=None)
    parser.add_argument("--ignore-overlap", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bypass-postprocessing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--postprocessing-yaml", type=Path, default=None)
    parser.add_argument("--verbose-report", action="store_true")
    parser.add_argument("--print-link-details", action="store_true")
    return parser.parse_args()


def disable_torch_compile(torch_module) -> None:
    if getattr(torch_module, "_sortformer_overlap_link_compile_disabled", False):
        return

    def _no_compile(model=None, *args, **kwargs):
        if model is None:
            return lambda fn: fn
        return model

    torch_module.compile = _no_compile
    torch_module._sortformer_overlap_link_compile_disabled = True


def force_cpu_cuda_state(torch_module) -> None:
    torch_module.cuda.is_available = lambda: False
    torch_module.cuda.current_device = lambda: None


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as src:
        for line in src:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def chunk_metadata(manifest: Path) -> dict[str, ChunkInfo]:
    metadata = {}
    for index, item in enumerate(read_jsonl(manifest)):
        chunk_id = str(item.get("uniq_id") or f"{Path(item['audio_filepath']).stem}_chunk{index:04d}")
        offset = float(item.get("offset") or 0.0)
        duration = float(item["duration"])
        metadata[chunk_id] = ChunkInfo(chunk_id=chunk_id, offset=offset, duration=duration)
    return metadata


def read_rttm_segments(rttm_dir: Path, metadata: dict[str, ChunkInfo], session_id: str) -> list[Segment]:
    segments = []
    for chunk_id in metadata:
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
                end = start + duration
                local_speaker = parts[7]
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
    return sorted(segments, key=lambda seg: (seg.start, seg.end, seg.chunk_id, seg.local_speaker))


def reference_speaker_count(rttm_path: Path) -> int:
    speakers = set()
    with rttm_path.open("r", encoding="utf-8") as src:
        for line in src:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == "SPEAKER":
                speakers.add(parts[7])
    return len(speakers)


def label_speaker_count(labels: list[str]) -> int:
    return len({label.strip().split()[2] for label in labels if len(label.strip().split()) >= 3})


def merge_intervals(intervals: list[tuple[float, float]], gap: float = 0.0) -> list[tuple[float, float]]:
    ordered = sorted((start, end) for start, end in intervals if end > start)
    if not ordered:
        return []
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + gap:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def interval_duration(intervals: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in intervals)


def interval_intersection(
    left: list[tuple[float, float]],
    right: list[tuple[float, float]],
) -> float:
    total = 0.0
    i = 0
    j = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if end > start:
            total += end - start
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def clipped_tracklet_intervals(
    segments: list[Segment],
    tracklet_id: str,
    overlap_start: float,
    overlap_end: float,
) -> list[tuple[float, float]]:
    intervals = []
    for segment in segments:
        if segment.tracklet_id != tracklet_id:
            continue
        start = max(segment.start, overlap_start)
        end = min(segment.end, overlap_end)
        if end > start:
            intervals.append((start, end))
    return merge_intervals(intervals)


def overlap_score(
    left: list[tuple[float, float]],
    right: list[tuple[float, float]],
) -> tuple[float, float]:
    left_duration = interval_duration(left)
    right_duration = interval_duration(right)
    if left_duration <= 0.0 or right_duration <= 0.0:
        return 0.0, 0.0
    intersection = interval_intersection(left, right)
    score = 2.0 * intersection / (left_duration + right_duration)
    return score, intersection


def link_tracklets(
    segments: list[Segment],
    chunks: dict[str, ChunkInfo],
    args: argparse.Namespace,
) -> dict[str, int]:
    by_chunk: dict[str, set[str]] = {}
    chunk_by_tracklet = {}
    for segment in segments:
        by_chunk.setdefault(segment.chunk_id, set()).add(segment.tracklet_id)
        chunk_by_tracklet[segment.tracklet_id] = segment.chunk_id

    tracklet_ids = sorted(chunk_by_tracklet)
    linker = UnionFind(tracklet_ids, chunk_by_tracklet)
    chunk_list = sorted(chunks.values(), key=lambda item: (item.offset, item.chunk_id))
    attempts = 0
    accepted = 0

    for left_pos, left_chunk in enumerate(chunk_list):
        for right_chunk in chunk_list[left_pos + 1 :]:
            if right_chunk.offset >= left_chunk.end:
                break
            overlap_start = max(left_chunk.offset, right_chunk.offset)
            overlap_end = min(left_chunk.end, right_chunk.end)
            if overlap_end - overlap_start < args.min_overlap_sec:
                continue

            candidates = []
            left_tracklets = sorted(by_chunk.get(left_chunk.chunk_id, set()))
            right_tracklets = sorted(by_chunk.get(right_chunk.chunk_id, set()))
            for left_tracklet in left_tracklets:
                left_intervals = clipped_tracklet_intervals(segments, left_tracklet, overlap_start, overlap_end)
                if not left_intervals:
                    continue
                for right_tracklet in right_tracklets:
                    right_intervals = clipped_tracklet_intervals(segments, right_tracklet, overlap_start, overlap_end)
                    if not right_intervals:
                        continue
                    score, intersection = overlap_score(left_intervals, right_intervals)
                    attempts += 1
                    if score >= args.link_threshold and intersection >= args.min_link_intersection:
                        candidates.append((score, intersection, left_tracklet, right_tracklet))

            used_left = set()
            used_right = set()
            for score, intersection, left_tracklet, right_tracklet in sorted(candidates, reverse=True):
                if left_tracklet in used_left or right_tracklet in used_right:
                    continue
                if not linker.can_union(left_tracklet, right_tracklet):
                    continue
                if linker.union(left_tracklet, right_tracklet):
                    accepted += 1
                    used_left.add(left_tracklet)
                    used_right.add(right_tracklet)
                    if args.print_link_details:
                        print(
                            "Link "
                            f"{left_tracklet} <-> {right_tracklet} "
                            f"score={score:.3f} shared={intersection:.3f}s"
                        )

    components = linker.components()
    component_order = sorted(
        components.values(),
        key=lambda items: min((chunks[chunk_by_tracklet[item]].offset, item) for item in items),
    )
    labels_by_tracklet = {}
    for label, items in enumerate(component_order):
        for item in items:
            labels_by_tracklet[item] = label

    print(
        "Overlap linking: "
        f"tracklets={len(tracklet_ids)} links={accepted} candidates_checked={attempts} "
        f"global_speakers={len(component_order)}"
    )
    return labels_by_tracklet


def write_global_rttm(
    segments: list[Segment],
    labels_by_tracklet: dict[str, int],
    output_path: Path,
    merge_gap: float,
) -> list[str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    by_speaker: dict[str, list[tuple[float, float]]] = {}
    session_id = segments[0].session_id if segments else "unknown"

    for segment in segments:
        if segment.tracklet_id not in labels_by_tracklet:
            continue
        speaker = f"speaker_{labels_by_tracklet[segment.tracklet_id]}"
        by_speaker.setdefault(speaker, []).append((segment.start, segment.end))

    label_rows = []
    rttm_rows = []
    for speaker in sorted(by_speaker):
        for start, end in merge_intervals(by_speaker[speaker], gap=merge_gap):
            duration = end - start
            if duration <= 0.0:
                continue
            label_rows.append(f"{start:.3f} {end:.3f} {speaker}")
            rttm_rows.append(
                f"SPEAKER {session_id} 1   {start:.3f}   {duration:.3f} "
                f"<NA> <NA> {speaker} <NA> <NA>\n"
            )

    rttm_rows.sort(key=lambda line: (float(line.split()[3]), float(line.split()[4]), line.split()[7]))
    label_rows.sort(key=lambda label: (float(label.split()[0]), float(label.split()[1]), label.split()[2]))
    with output_path.open("w", encoding="utf-8") as dst:
        dst.writelines(rttm_rows)
    return label_rows


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
    chunks = chunk_metadata(uniq_manifest)
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

    segments = read_rttm_segments(sortformer_rttm_dir, chunks, args.session_id)
    if not segments:
        raise RuntimeError(f"No Sortformer segments found in {sortformer_rttm_dir}")
    print(f"Sortformer local segments: {len(segments)}")

    labels_by_tracklet = link_tracklets(segments, chunks, args)
    global_rttm_path = global_rttm_dir / f"{args.session_id}.rttm"
    hyp_labels = write_global_rttm(segments, labels_by_tracklet, global_rttm_path, merge_gap=args.merge_gap)
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
