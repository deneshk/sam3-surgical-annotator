#!/usr/bin/env python
"""Ad hoc evaluation script for chunked single-object tracker memory recall."""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from PIL import ImageDraw


SCRIPT_PATH = Path(__file__).resolve()
APP_ROOT = SCRIPT_PATH.parents[1]
SRC_ROOT = APP_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from annotator.vendor.sam3_runtime import ensure_vendor_sam3_on_path

ensure_vendor_sam3_on_path()


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
TEMPLATE_SIZE = 48
REACQUIRE_BOX_COLOR = (255, 215, 0)
SEED_BOX_COLOR = (0, 255, 0)


@dataclass
class FrameRecord:
    frame_idx: int
    score: float
    area: int
    box_xyxy: list[float]
    source_name: str
    mask: torch.Tensor


@dataclass
class AnchorRecord:
    frame_idx: int
    score: float
    area: int
    source_name: str
    box_xyxy: list[float]
    template: torch.Tensor
    template_weight: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Chunked single-object SAM3 harness that carries lightweight appearance "
            "memory across resets, reacquires the object, and re-prompts SAM3."
        )
    )
    parser.add_argument("--image-dir", required=True, help="Directory of ordered frame images.")
    parser.add_argument(
        "--start-frame",
        type=int,
        required=True,
        help="Inclusive start frame index in the naturally sorted directory listing.",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        required=True,
        help="Inclusive end frame index in the naturally sorted directory listing.",
    )
    parser.add_argument(
        "--box",
        type=float,
        nargs=4,
        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
        required=True,
        help="Seed box in source-image pixel xyxy coordinates on the start frame.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        required=True,
        help="Number of frames to process before intentionally resetting the tracker state.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=os.getenv("SAM3_CHECKPOINT_PATH"),
        help="Path to the SAM3 checkpoint. Falls back to SAM3_CHECKPOINT_PATH if set.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where a session output folder will be created.",
    )
    parser.add_argument("--obj-id", type=int, default=1, help="Single tracked object id.")
    parser.add_argument(
        "--memory-mode",
        choices=("native", "reacquire"),
        default="reacquire",
        help=(
            "Use native SAM3 session memory only, or the external reacquire-and-reprompt "
            "flow across chunk resets."
        ),
    )
    parser.add_argument("--image-size", type=int, default=1008, help="SAM3 image size.")
    parser.add_argument(
        "--max-anchors",
        type=int,
        default=5,
        help="Maximum number of anchor memories to keep across chunk resets.",
    )
    parser.add_argument(
        "--lost-threshold",
        type=float,
        default=0.0,
        help="Treat the object as lost if its score is <= threshold or the mask is empty.",
    )
    parser.add_argument(
        "--recover-within-chunk",
        action="store_true",
        help="Attempt recovery inside a chunk when a frame is detected as lost.",
    )
    parser.add_argument(
        "--offload-state-to-cpu",
        action="store_true",
        help="Store tracker state tensors on CPU where supported to reduce GPU memory pressure.",
    )
    parser.add_argument(
        "--offload-video-to-cpu",
        action="store_true",
        help="Keep decoded video frames on CPU inside the tracker state.",
    )
    parser.add_argument(
        "--save-anchor-debug",
        action="store_true",
        help="Persist per-anchor .pt snapshots for inspection.",
    )
    parser.add_argument(
        "--reacquire-score-threshold",
        type=float,
        default=0.42,
        help="Minimum appearance score needed to accept a reacquired box.",
    )
    parser.add_argument(
        "--reacquire-search-size",
        type=int,
        default=640,
        help="Resize long image side to this value during reacquisition search.",
    )
    parser.add_argument(
        "--reacquire-stride-factor",
        type=float,
        default=0.5,
        help="Grid stride as a fraction of the candidate box size during reacquisition.",
    )
    parser.add_argument(
        "--reacquire-scales",
        default="0.9,1.0,1.1",
        help="Comma-separated candidate box scales for reacquisition search.",
    )
    return parser.parse_args()


def natural_key(value: str) -> list[Any]:
    parts = re.split(r"(\d+)", value)
    key: list[Any] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def list_frame_paths(image_dir: Path) -> list[Path]:
    frame_paths = [
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]
    frame_paths.sort(key=lambda path: natural_key(path.name))
    return frame_paths


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_session_dir(base_dir: Path) -> Path:
    ensure_dir(base_dir)
    session_dir = base_dir / f"session_{time.strftime('%Y%m%d_%H%M%S')}"
    ensure_dir(session_dir)
    ensure_dir(session_dir / "masks")
    ensure_dir(session_dir / "anchors")
    return session_dir


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def load_checkpoint_weights(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unsupported checkpoint format in {checkpoint_path}")
    return ckpt


def load_tracker_weights(
    tracker: torch.nn.Module, checkpoint_path: Path
) -> tuple[list[str], list[str]]:
    ckpt = load_checkpoint_weights(checkpoint_path)
    tracker_keys = {k[len("tracker.") :]: v for k, v in ckpt.items() if k.startswith("tracker.")}
    detector_backbone_keys = {
        k.replace("detector.backbone.", "backbone."): v
        for k, v in ckpt.items()
        if k.startswith("detector.backbone.")
    }
    state_dict = {}
    state_dict.update(tracker_keys)
    state_dict.update(detector_backbone_keys)
    if not state_dict:
        state_dict = ckpt
    missing_keys, unexpected_keys = tracker.load_state_dict(state_dict, strict=False)
    return list(missing_keys), list(unexpected_keys)


def build_tracker_model(args: argparse.Namespace) -> tuple[torch.nn.Module, torch.device]:
    from sam3.model_builder import build_tracker

    if not args.checkpoint_path:
        raise RuntimeError(
            "A checkpoint path is required. Pass --checkpoint-path or set SAM3_CHECKPOINT_PATH."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tracker = build_tracker(
        apply_temporal_disambiguation=True,
        with_backbone=True,
        image_size=args.image_size,
    )
    missing_keys, unexpected_keys = load_tracker_weights(
        tracker, Path(args.checkpoint_path)
    )
    tracker.to(device=device)
    tracker.eval()
    if missing_keys:
        print(f"Missing tracker keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected tracker keys: {unexpected_keys}")
    return tracker, device


def empty_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def create_tracker_state(
    tracker: torch.nn.Module,
    image_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return tracker.init_state(
        video_path=str(image_dir),
        offload_video_to_cpu=args.offload_video_to_cpu,
        offload_state_to_cpu=args.offload_state_to_cpu,
        async_loading_frames=False,
    )


def stage_frame_subset(
    frame_paths: list[Path],
    session_dir: Path,
) -> tuple[Path, list[dict[str, Any]]]:
    staged_dir = session_dir / "staged_frames"
    ensure_dir(staged_dir)
    mapping: list[dict[str, Any]] = []
    for local_idx, frame_path in enumerate(frame_paths):
        staged_name = f"{local_idx:06d}.jpg"
        staged_path = staged_dir / staged_name
        with Image.open(frame_path) as image:
            rgb_image = image.convert("RGB")
            rgb_image.save(staged_path, format="JPEG", quality=95)
        mapping.append(
            {
                "local_idx": local_idx,
                "source_name": frame_path.name,
                "staged_name": staged_name,
            }
        )
    return staged_dir, mapping


def scale_box_to_model_space(
    box_xyxy: list[float],
    frame_path: Path,
    image_size: int,
) -> list[float]:
    with Image.open(frame_path) as image:
        width, height = image.size
    x_scale = image_size / float(width)
    y_scale = image_size / float(height)
    xmin, ymin, xmax, ymax = box_xyxy
    return [xmin * x_scale, ymin * y_scale, xmax * x_scale, ymax * y_scale]


def clamp_box(box_xyxy: list[float], width: int, height: int) -> list[float]:
    xmin, ymin, xmax, ymax = box_xyxy
    xmin = max(0.0, min(xmin, width - 1))
    ymin = max(0.0, min(ymin, height - 1))
    xmax = max(xmin + 1.0, min(xmax, width))
    ymax = max(ymin + 1.0, min(ymax, height))
    return [xmin, ymin, xmax, ymax]


def mask_to_box(mask: torch.Tensor) -> list[float] | None:
    ys, xs = torch.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    xmin = int(xs.min().item())
    xmax = int(xs.max().item()) + 1
    ymin = int(ys.min().item())
    ymax = int(ys.max().item()) + 1
    return [float(xmin), float(ymin), float(xmax), float(ymax)]


def frame_area_from_mask(mask: torch.Tensor | None) -> int:
    if mask is None:
        return 0
    return int((mask > 0).sum().item())


def score_from_output(out: dict[str, Any]) -> float:
    score_tensor = out.get("object_score_logits")
    if score_tensor is None:
        return float("-inf")
    return float(score_tensor.reshape(-1)[0].detach().cpu().item())


def parse_scales(raw: str) -> list[float]:
    scales = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not scales:
        raise RuntimeError("Expected at least one value in --reacquire-scales.")
    return scales


def load_rgb_image(frame_path: Path) -> Image.Image:
    with Image.open(frame_path) as image:
        return image.convert("RGB")


def save_plain_frame(frame_path: Path, output_path: Path) -> None:
    image = load_rgb_image(frame_path)
    image.save(output_path)


def save_overlay(
    frame_path: Path,
    mask: torch.Tensor | None,
    path: Path,
    mask_color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.4,
    prompt_box_xyxy: list[float] | None = None,
    prompt_box_color: tuple[int, int, int] = SEED_BOX_COLOR,
) -> int:
    with Image.open(frame_path) as image:
        base = image.convert("RGB")
    if mask is None:
        output = base
        mask_area = 0
    else:
        mask_bool = mask.detach().cpu().numpy().astype(bool)
        base_np = np.asarray(base).astype(np.float32)
        overlay_np = base_np.copy()
        color_np = np.asarray(mask_color, dtype=np.float32)
        overlay_np[mask_bool] = (
            (1.0 - alpha) * overlay_np[mask_bool] + alpha * color_np
        )
        output = Image.fromarray(np.clip(overlay_np, 0, 255).astype(np.uint8), mode="RGB")
        mask_area = int(mask_bool.sum())
    if prompt_box_xyxy is not None:
        draw = ImageDraw.Draw(output)
        draw.rectangle(prompt_box_xyxy, outline=prompt_box_color, width=3)
    output.save(path)
    return mask_area


def crop_array(array: np.ndarray, box_xyxy: list[float]) -> np.ndarray:
    height, width = array.shape[:2]
    xmin, ymin, xmax, ymax = clamp_box(box_xyxy, width, height)
    return array[int(ymin) : int(ymax), int(xmin) : int(xmax)]


def build_template_from_mask(
    frame_path: Path,
    mask: torch.Tensor,
    box_xyxy: list[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    frame_rgb = np.asarray(load_rgb_image(frame_path), dtype=np.float32) / 255.0
    mask_np = mask.detach().cpu().numpy().astype(np.float32)
    crop_rgb = crop_array(frame_rgb, box_xyxy)
    crop_mask = crop_array(mask_np, box_xyxy)
    if crop_rgb.size == 0 or crop_mask.size == 0:
        raise RuntimeError(f"Empty crop while building template for {frame_path}")
    crop_rgb_img = Image.fromarray(np.clip(crop_rgb * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    crop_mask_img = Image.fromarray(np.clip(crop_mask * 255.0, 0, 255).astype(np.uint8), mode="L")
    crop_rgb_img = crop_rgb_img.resize((TEMPLATE_SIZE, TEMPLATE_SIZE), resample=Image.BILINEAR)
    crop_mask_img = crop_mask_img.resize((TEMPLATE_SIZE, TEMPLATE_SIZE), resample=Image.BILINEAR)
    template = torch.from_numpy(np.asarray(crop_rgb_img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    weight = torch.from_numpy(np.asarray(crop_mask_img, dtype=np.float32) / 255.0).unsqueeze(0)
    weight = torch.clamp(weight, min=0.0, max=1.0)
    if float(weight.sum().item()) <= 0.0:
        weight = torch.ones_like(weight)
    return template, weight


def build_anchor_bank(
    frame_records: dict[int, FrameRecord],
    frame_paths: list[Path],
    max_anchors: int,
) -> list[AnchorRecord]:
    candidates = [
        record
        for record in frame_records.values()
        if record.area > 0 and record.score > 0 and record.box_xyxy is not None
    ]
    if not candidates:
        return []
    candidates.sort(key=lambda item: (item.frame_idx, item.score))
    seed = candidates[0]
    most_recent = max(candidates, key=lambda item: item.frame_idx)
    by_quality = sorted(
        candidates,
        key=lambda item: (item.score, item.area, item.frame_idx),
        reverse=True,
    )
    selected: list[FrameRecord] = [seed]
    if most_recent.frame_idx != seed.frame_idx:
        selected.append(most_recent)
    for item in by_quality:
        if len(selected) >= max_anchors:
            break
        if any(existing.frame_idx == item.frame_idx for existing in selected):
            continue
        selected.append(item)
    selected.sort(key=lambda item: item.frame_idx)
    anchors: list[AnchorRecord] = []
    for record in selected[:max_anchors]:
        template, weight = build_template_from_mask(
            frame_paths[record.frame_idx],
            record.mask,
            record.box_xyxy,
        )
        anchors.append(
            AnchorRecord(
                frame_idx=record.frame_idx,
                score=record.score,
                area=record.area,
                source_name=record.source_name,
                box_xyxy=record.box_xyxy,
                template=template,
                template_weight=weight,
            )
        )
    return anchors


def persist_anchor_bank(
    anchor_bank: list[AnchorRecord],
    anchor_dir: Path,
    chunk_index: int,
    save_anchor_debug: bool,
) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for anchor in anchor_bank:
        filename = f"chunk_{chunk_index:03d}_frame_{anchor.frame_idx:06d}.pt"
        anchor_path = anchor_dir / filename
        torch.save(
            {
                "frame_idx": anchor.frame_idx,
                "score": anchor.score,
                "area": anchor.area,
                "source_name": anchor.source_name,
                "box_xyxy": anchor.box_xyxy,
                "template": anchor.template,
                "template_weight": anchor.template_weight,
            },
            anchor_path,
        )
        serialized.append(
            {
                "frame_idx": anchor.frame_idx,
                "score": anchor.score,
                "area": anchor.area,
                "source_name": anchor.source_name,
                "box_xyxy": anchor.box_xyxy,
                "anchor_file": anchor_path.name,
            }
        )
    if save_anchor_debug:
        write_json(anchor_dir / f"bank_after_chunk_{chunk_index:03d}.json", {"anchors": serialized})
    return serialized


def load_anchor_bank(anchor_dir: Path, metadata: list[dict[str, Any]]) -> list[AnchorRecord]:
    anchors: list[AnchorRecord] = []
    for item in metadata:
        data = torch.load(anchor_dir / item["anchor_file"], map_location="cpu", weights_only=False)
        anchors.append(
            AnchorRecord(
                frame_idx=int(data["frame_idx"]),
                score=float(data["score"]),
                area=int(data["area"]),
                source_name=str(data["source_name"]),
                box_xyxy=[float(x) for x in data["box_xyxy"]],
                template=data["template"].float().cpu(),
                template_weight=data["template_weight"].float().cpu(),
            )
        )
    anchors.sort(key=lambda item: item.frame_idx)
    return anchors


def candidate_similarity(
    frame_np: np.ndarray,
    candidate_box_xyxy: list[float],
    anchor: AnchorRecord,
) -> float:
    candidate_crop = crop_array(frame_np, candidate_box_xyxy)
    if candidate_crop.size == 0:
        return float("-inf")
    crop_img = Image.fromarray(candidate_crop, mode="RGB").resize(
        (TEMPLATE_SIZE, TEMPLATE_SIZE),
        resample=Image.BILINEAR,
    )
    crop_tensor = torch.from_numpy(np.asarray(crop_img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    weight = anchor.template_weight
    crop_vec = (crop_tensor * weight).reshape(-1)
    templ_vec = (anchor.template * weight).reshape(-1)
    crop_norm = torch.linalg.norm(crop_vec)
    templ_norm = torch.linalg.norm(templ_vec)
    if float(crop_norm.item()) == 0.0 or float(templ_norm.item()) == 0.0:
        return float("-inf")
    score = torch.dot(crop_vec, templ_vec) / (crop_norm * templ_norm)
    return float(score.item())


def resized_image_and_scale(frame_path: Path, search_size: int) -> tuple[np.ndarray, float]:
    image = load_rgb_image(frame_path)
    width, height = image.size
    scale = min(1.0, float(search_size) / float(max(width, height)))
    if scale < 1.0:
        resized = image.resize((int(round(width * scale)), int(round(height * scale))), resample=Image.BILINEAR)
    else:
        resized = image
    return np.asarray(resized, dtype=np.uint8), scale


def search_frame_for_reacquire_box(
    frame_path: Path,
    anchor_bank: list[AnchorRecord],
    scales: list[float],
    search_size: int,
    stride_factor: float,
) -> tuple[list[float], float, AnchorRecord] | None:
    frame_np, resize_scale = resized_image_and_scale(frame_path, search_size)
    height, width = frame_np.shape[:2]
    best_score = float("-inf")
    best_box: list[float] | None = None
    best_anchor: AnchorRecord | None = None
    ranked_anchors = sorted(anchor_bank, key=lambda item: (item.frame_idx, item.score), reverse=True)[:3]
    for anchor in ranked_anchors:
        base_w = max(8, int(round((anchor.box_xyxy[2] - anchor.box_xyxy[0]) * resize_scale)))
        base_h = max(8, int(round((anchor.box_xyxy[3] - anchor.box_xyxy[1]) * resize_scale)))
        for scale in scales:
            cand_w = max(8, int(round(base_w * scale)))
            cand_h = max(8, int(round(base_h * scale)))
            step_x = max(8, int(round(cand_w * stride_factor)))
            step_y = max(8, int(round(cand_h * stride_factor)))
            max_x = max(0, width - cand_w)
            max_y = max(0, height - cand_h)
            x_positions = list(range(0, max_x + 1, step_x))
            y_positions = list(range(0, max_y + 1, step_y))
            if not x_positions or x_positions[-1] != max_x:
                x_positions.append(max_x)
            if not y_positions or y_positions[-1] != max_y:
                y_positions.append(max_y)
            for y in y_positions:
                for x in x_positions:
                    score = candidate_similarity(
                        frame_np,
                        [float(x), float(y), float(x + cand_w), float(y + cand_h)],
                        anchor,
                    )
                    if score > best_score:
                        best_score = score
                        best_box = [
                            float(x) / resize_scale,
                            float(y) / resize_scale,
                            float(x + cand_w) / resize_scale,
                            float(y + cand_h) / resize_scale,
                        ]
                        best_anchor = anchor
    if best_box is None or best_anchor is None:
        return None
    return best_box, best_score, best_anchor


def find_reacquisition(
    frame_paths: list[Path],
    chunk_start: int,
    chunk_end: int,
    anchor_bank: list[AnchorRecord],
    args: argparse.Namespace,
    events_path: Path,
    chunk_index: int,
    global_start_frame: int,
) -> tuple[int, list[float], float] | None:
    scales = parse_scales(args.reacquire_scales)
    for frame_idx in range(chunk_start, chunk_end + 1):
        result = search_frame_for_reacquire_box(
            frame_paths[frame_idx],
            anchor_bank,
            scales=scales,
            search_size=args.reacquire_search_size,
            stride_factor=args.reacquire_stride_factor,
        )
        if result is None:
            continue
        box_xyxy, score, source_anchor = result
        append_jsonl(
            events_path,
            {
                "event": "reacquire_scan",
                "chunk_index": chunk_index,
                "frame_idx": global_start_frame + frame_idx,
                "score": score,
                "source_anchor_frame": global_start_frame + source_anchor.frame_idx,
            },
        )
        if score >= args.reacquire_score_threshold:
            return frame_idx, box_xyxy, score
    return None


def seed_with_box(
    tracker: torch.nn.Module,
    tracker_state: dict[str, Any],
    frame_idx: int,
    obj_id: int,
    box_xyxy: list[float],
    frame_path: Path,
    image_size: int,
) -> None:
    tracker.add_new_points_or_box(
        inference_state=tracker_state,
        frame_idx=frame_idx,
        obj_id=obj_id,
        box=scale_box_to_model_space(box_xyxy, frame_path, image_size),
        clear_old_points=True,
        rel_coordinates=False,
    )


def record_frame_output(
    frame_records: dict[int, FrameRecord],
    frame_idx: int,
    score: float,
    mask_area: int,
    box_xyxy: list[float] | None,
    source_name: str,
    mask: torch.Tensor,
) -> None:
    if box_xyxy is None:
        return
    frame_records[frame_idx] = FrameRecord(
        frame_idx=frame_idx,
        score=score,
        area=mask_area,
        box_xyxy=box_xyxy,
        source_name=source_name,
        mask=mask.clone(),
    )


def is_lost(mask_area: int, score: float, lost_threshold: float) -> bool:
    return mask_area <= 0 or score <= lost_threshold


def chunk_ranges(start_frame: int, end_frame: int, chunk_size: int) -> list[tuple[int, int]]:
    ranges = []
    current = start_frame
    while current <= end_frame:
        chunk_end = min(current + chunk_size - 1, end_frame)
        ranges.append((current, chunk_end))
        current = chunk_end + 1
    return ranges


def run_propagation(
    tracker: torch.nn.Module,
    tracker_state: dict[str, Any],
    frame_paths: list[Path],
    start_frame_idx: int,
    chunk_end: int,
    args: argparse.Namespace,
    masks_dir: Path,
    events_path: Path,
    summary: dict[str, Any],
    frame_records: dict[int, FrameRecord],
    global_start_frame: int,
    prompt_box_by_frame: dict[int, tuple[list[float], tuple[int, int, int]]],
) -> int | None:
    iterator = tracker.propagate_in_video(
        inference_state=tracker_state,
        start_frame_idx=start_frame_idx,
        max_frame_num_to_track=chunk_end - start_frame_idx,
        reverse=False,
        tqdm_disable=False,
        propagate_preflight=True,
    )
    for frame_idx, obj_ids, _low_res_masks, video_res_masks, obj_scores in iterator:
        if frame_idx < start_frame_idx or frame_idx > chunk_end:
            continue
        global_frame_idx = global_start_frame + frame_idx
        if len(obj_ids) == 0 or args.obj_id not in list(obj_ids):
            save_plain_frame(frame_paths[frame_idx], masks_dir / f"frame_{global_frame_idx:06d}.png")
            summary["frames_lost"].append(global_frame_idx)
            append_jsonl(
                events_path,
                {
                    "event": "empty_output",
                    "frame_idx": global_frame_idx,
                },
            )
            return frame_idx if args.recover_within_chunk and frame_idx < chunk_end else None

        obj_index = list(obj_ids).index(args.obj_id)
        mask = (video_res_masks[obj_index, 0] > 0).detach().cpu()
        score = float(obj_scores[obj_index].detach().cpu().item())
        box_xyxy = mask_to_box(mask)
        overlay_path = masks_dir / f"frame_{global_frame_idx:06d}.png"
        prompt_entry = prompt_box_by_frame.get(frame_idx)
        prompt_box = prompt_entry[0] if prompt_entry is not None else None
        prompt_color = prompt_entry[1] if prompt_entry is not None else SEED_BOX_COLOR
        mask_area = save_overlay(
            frame_paths[frame_idx],
            mask,
            overlay_path,
            prompt_box_xyxy=prompt_box,
            prompt_box_color=prompt_color,
        )
        summary["frames_tracked"].append(global_frame_idx)
        append_jsonl(
            events_path,
            {
                "event": "frame_output",
                "frame_idx": global_frame_idx,
                "mask_file": overlay_path.name,
                "mask_area": mask_area,
                "score": score,
                "source_name": frame_paths[frame_idx].name,
            },
        )
        if box_xyxy is not None:
            record_frame_output(
                frame_records=frame_records,
                frame_idx=frame_idx,
                score=score,
                mask_area=mask_area,
                box_xyxy=box_xyxy,
                source_name=frame_paths[frame_idx].name,
                mask=mask,
            )
        if is_lost(mask_area, score, args.lost_threshold):
            summary["frames_lost"].append(global_frame_idx)
            append_jsonl(
                events_path,
                {
                    "event": "lost",
                    "frame_idx": global_frame_idx,
                    "mask_area": mask_area,
                    "score": score,
                },
            )
            return frame_idx if args.recover_within_chunk and frame_idx < chunk_end else None
    return None


def run_native_propagation(
    tracker: torch.nn.Module,
    tracker_state: dict[str, Any],
    frame_paths: list[Path],
    args: argparse.Namespace,
    masks_dir: Path,
    events_path: Path,
    summary: dict[str, Any],
    frame_records: dict[int, FrameRecord],
    global_start_frame: int,
    prompt_box_by_frame: dict[int, tuple[list[float], tuple[int, int, int]]],
) -> None:
    iterator = tracker.propagate_in_video(
        inference_state=tracker_state,
        start_frame_idx=0,
        max_frame_num_to_track=len(frame_paths) - 1,
        reverse=False,
        tqdm_disable=False,
        propagate_preflight=True,
    )
    for frame_idx, obj_ids, _low_res_masks, video_res_masks, obj_scores in iterator:
        global_frame_idx = global_start_frame + frame_idx
        if len(obj_ids) == 0 or args.obj_id not in list(obj_ids):
            save_plain_frame(frame_paths[frame_idx], masks_dir / f"frame_{global_frame_idx:06d}.png")
            summary["frames_lost"].append(global_frame_idx)
            append_jsonl(
                events_path,
                {
                    "event": "empty_output",
                    "frame_idx": global_frame_idx,
                },
            )
            continue

        obj_index = list(obj_ids).index(args.obj_id)
        mask = (video_res_masks[obj_index, 0] > 0).detach().cpu()
        score = float(obj_scores[obj_index].detach().cpu().item())
        box_xyxy = mask_to_box(mask)
        overlay_path = masks_dir / f"frame_{global_frame_idx:06d}.png"
        prompt_entry = prompt_box_by_frame.get(frame_idx)
        prompt_box = prompt_entry[0] if prompt_entry is not None else None
        prompt_color = prompt_entry[1] if prompt_entry is not None else SEED_BOX_COLOR
        mask_area = save_overlay(
            frame_paths[frame_idx],
            mask,
            overlay_path,
            prompt_box_xyxy=prompt_box,
            prompt_box_color=prompt_color,
        )
        summary["frames_tracked"].append(global_frame_idx)
        append_jsonl(
            events_path,
            {
                "event": "frame_output",
                "frame_idx": global_frame_idx,
                "mask_file": overlay_path.name,
                "mask_area": mask_area,
                "score": score,
                "source_name": frame_paths[frame_idx].name,
            },
        )
        if box_xyxy is not None:
            record_frame_output(
                frame_records=frame_records,
                frame_idx=frame_idx,
                score=score,
                mask_area=mask_area,
                box_xyxy=box_xyxy,
                source_name=frame_paths[frame_idx].name,
                mask=mask,
            )
        if is_lost(mask_area, score, args.lost_threshold):
            summary["frames_lost"].append(global_frame_idx)
            append_jsonl(
                events_path,
                {
                    "event": "lost",
                    "frame_idx": global_frame_idx,
                    "mask_area": mask_area,
                    "score": score,
                },
            )


def save_untracked_frames(
    frame_paths: list[Path],
    start_idx: int,
    end_idx: int,
    masks_dir: Path,
    events_path: Path,
    global_start_frame: int,
) -> None:
    for frame_idx in range(start_idx, end_idx + 1):
        global_frame_idx = global_start_frame + frame_idx
        output_path = masks_dir / f"frame_{global_frame_idx:06d}.png"
        if output_path.exists():
            continue
        save_plain_frame(frame_paths[frame_idx], output_path)
        append_jsonl(
            events_path,
            {
                "event": "untracked_frame",
                "frame_idx": global_frame_idx,
                "mask_file": output_path.name,
            },
        )


def run_native_session(
    tracker: torch.nn.Module,
    staged_dir: Path,
    selected_frame_paths: list[Path],
    args: argparse.Namespace,
    session_dir: Path,
    events_path: Path,
    summary: dict[str, Any],
) -> None:
    tracker_state = create_tracker_state(tracker, staged_dir, args)
    seed_with_box(
        tracker=tracker,
        tracker_state=tracker_state,
        frame_idx=0,
        obj_id=args.obj_id,
        box_xyxy=list(args.box),
        frame_path=selected_frame_paths[0],
        image_size=args.image_size,
    )
    prompt_box_by_frame = {0: (list(args.box), SEED_BOX_COLOR)}
    frame_records: dict[int, FrameRecord] = {}
    run_native_propagation(
        tracker=tracker,
        tracker_state=tracker_state,
        frame_paths=selected_frame_paths,
        args=args,
        masks_dir=session_dir / "masks",
        events_path=events_path,
        summary=summary,
        frame_records=frame_records,
        global_start_frame=args.start_frame,
        prompt_box_by_frame=prompt_box_by_frame,
    )
    anchor_bank = build_anchor_bank(
        frame_records=frame_records,
        frame_paths=selected_frame_paths,
        max_anchors=args.max_anchors,
    )
    if anchor_bank:
        summary["anchor_bank"] = persist_anchor_bank(
            anchor_bank=anchor_bank,
            anchor_dir=session_dir / "anchors",
            chunk_index=0,
            save_anchor_debug=args.save_anchor_debug,
        )
    del tracker_state
    empty_cuda_cache()


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir).resolve()
    if not image_dir.is_dir():
        raise RuntimeError(f"Image directory does not exist: {image_dir}")

    frame_paths = list_frame_paths(image_dir)
    if not frame_paths:
        raise RuntimeError(f"No image frames found in {image_dir}")

    if args.start_frame < 0 or args.end_frame >= len(frame_paths) or args.start_frame > args.end_frame:
        raise RuntimeError(
            f"Invalid frame range [{args.start_frame}, {args.end_frame}] for {len(frame_paths)} frames."
        )

    if args.chunk_size <= 0:
        raise RuntimeError("--chunk-size must be positive.")

    session_dir = get_session_dir(Path(args.output_dir).resolve())
    events_path = session_dir / "events.jsonl"
    tracker, device = build_tracker_model(args)

    selected_frame_paths = frame_paths[args.start_frame : args.end_frame + 1]
    staged_dir, staged_mapping = stage_frame_subset(selected_frame_paths, session_dir)
    local_start_frame = 0
    local_end_frame = len(selected_frame_paths) - 1
    chunk_plan = chunk_ranges(local_start_frame, local_end_frame, args.chunk_size)
    run_config = {
        "image_dir": str(image_dir),
        "start_frame": args.start_frame,
        "end_frame": args.end_frame,
        "box_xyxy": [float(x) for x in args.box],
        "chunk_size": args.chunk_size,
        "checkpoint_path": args.checkpoint_path,
        "obj_id": args.obj_id,
        "memory_mode": args.memory_mode,
        "image_size": args.image_size,
        "max_anchors": args.max_anchors,
        "lost_threshold": args.lost_threshold,
        "recover_within_chunk": args.recover_within_chunk,
        "reacquire_score_threshold": args.reacquire_score_threshold,
        "reacquire_search_size": args.reacquire_search_size,
        "reacquire_stride_factor": args.reacquire_stride_factor,
        "reacquire_scales": parse_scales(args.reacquire_scales),
        "offload_state_to_cpu": args.offload_state_to_cpu,
        "offload_video_to_cpu": args.offload_video_to_cpu,
        "device": str(device),
        "staged_dir": str(staged_dir),
    }
    write_json(session_dir / "run_config.json", run_config)
    write_json(session_dir / "frame_mapping.json", {"frames": staged_mapping})

    summary: dict[str, Any] = {
        "image_dir": str(image_dir),
        "start_frame": args.start_frame,
        "end_frame": args.end_frame,
        "chunk_size": args.chunk_size,
        "num_chunks": len(chunk_plan),
        "frames_tracked": [],
        "frames_lost": [],
        "chunk_resets": 0,
        "chunk_recoveries": [],
        "within_chunk_recoveries": [],
        "anchor_bank": [],
    }

    if args.memory_mode == "native":
        run_native_session(
            tracker=tracker,
            staged_dir=staged_dir,
            selected_frame_paths=selected_frame_paths,
            args=args,
            session_dir=session_dir,
            events_path=events_path,
            summary=summary,
        )
        summary["frames_tracked"] = sorted(set(summary["frames_tracked"]))
        summary["frames_lost"] = sorted(set(summary["frames_lost"]))
        write_json(session_dir / "summary.json", summary)
        print(f"Session written to: {session_dir}")
        return

    anchor_bank: list[AnchorRecord] = []
    anchor_metadata: list[dict[str, Any]] = []
    frame_records: dict[int, FrameRecord] = {}

    for chunk_index, (chunk_start, chunk_end) in enumerate(chunk_plan):
        chunk_global_start = args.start_frame + chunk_start
        chunk_global_end = args.start_frame + chunk_end
        append_jsonl(
            events_path,
            {
                "event": "chunk_start",
                "chunk_index": chunk_index,
                "chunk_start": chunk_global_start,
                "chunk_end": chunk_global_end,
            },
        )

        prompt_box_by_frame: dict[int, tuple[list[float], tuple[int, int, int]]] = {}
        propagate_start: int | None = None
        tracker_state = create_tracker_state(tracker, staged_dir, args)
        if chunk_index == 0:
            seed_with_box(
                tracker=tracker,
                tracker_state=tracker_state,
                frame_idx=chunk_start,
                obj_id=args.obj_id,
                box_xyxy=list(args.box),
                frame_path=selected_frame_paths[chunk_start],
                image_size=args.image_size,
            )
            prompt_box_by_frame[chunk_start] = (list(args.box), SEED_BOX_COLOR)
            propagate_start = chunk_start
        else:
            summary["chunk_resets"] += 1
            anchor_bank = load_anchor_bank(session_dir / "anchors", anchor_metadata)
            reacquired = find_reacquisition(
                frame_paths=selected_frame_paths,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                anchor_bank=anchor_bank,
                args=args,
                events_path=events_path,
                chunk_index=chunk_index,
                global_start_frame=args.start_frame,
            )
            if reacquired is not None:
                reacquire_frame_idx, reacquire_box, reacquire_score = reacquired
                seed_with_box(
                    tracker=tracker,
                    tracker_state=tracker_state,
                    frame_idx=reacquire_frame_idx,
                    obj_id=args.obj_id,
                    box_xyxy=reacquire_box,
                    frame_path=selected_frame_paths[reacquire_frame_idx],
                    image_size=args.image_size,
                )
                prompt_box_by_frame[reacquire_frame_idx] = (reacquire_box, REACQUIRE_BOX_COLOR)
                propagate_start = reacquire_frame_idx
                summary["chunk_recoveries"].append(
                    {
                        "chunk_index": chunk_index,
                        "frame_idx": args.start_frame + reacquire_frame_idx,
                        "score": reacquire_score,
                        "box_xyxy": reacquire_box,
                    }
                )
                append_jsonl(
                    events_path,
                    {
                        "event": "chunk_reacquired",
                        "chunk_index": chunk_index,
                        "frame_idx": args.start_frame + reacquire_frame_idx,
                        "score": reacquire_score,
                        "box_xyxy": reacquire_box,
                    },
                )
                if reacquire_frame_idx > chunk_start:
                    save_untracked_frames(
                        frame_paths=selected_frame_paths,
                        start_idx=chunk_start,
                        end_idx=reacquire_frame_idx - 1,
                        masks_dir=session_dir / "masks",
                        events_path=events_path,
                        global_start_frame=args.start_frame,
                    )
            else:
                append_jsonl(
                    events_path,
                    {
                        "event": "chunk_reacquire_failed",
                        "chunk_index": chunk_index,
                        "chunk_start": chunk_global_start,
                        "chunk_end": chunk_global_end,
                    },
                )
                save_untracked_frames(
                    frame_paths=selected_frame_paths,
                    start_idx=chunk_start,
                    end_idx=chunk_end,
                    masks_dir=session_dir / "masks",
                    events_path=events_path,
                    global_start_frame=args.start_frame,
                )

        while propagate_start is not None and propagate_start <= chunk_end:
            lost_frame = run_propagation(
                tracker=tracker,
                tracker_state=tracker_state,
                frame_paths=selected_frame_paths,
                start_frame_idx=propagate_start,
                chunk_end=chunk_end,
                args=args,
                masks_dir=session_dir / "masks",
                events_path=events_path,
                summary=summary,
                frame_records=frame_records,
                global_start_frame=args.start_frame,
                prompt_box_by_frame=prompt_box_by_frame,
            )
            if lost_frame is None or not args.recover_within_chunk:
                break

            anchor_bank = build_anchor_bank(
                frame_records=frame_records,
                frame_paths=selected_frame_paths,
                max_anchors=args.max_anchors,
            )
            if not anchor_bank:
                append_jsonl(
                    events_path,
                    {
                        "event": "reacquire_failed",
                        "chunk_index": chunk_index,
                        "frame_idx": args.start_frame + lost_frame,
                        "reason": "empty_anchor_bank",
                    },
                )
                break
            anchor_metadata = persist_anchor_bank(
                anchor_bank=anchor_bank,
                anchor_dir=session_dir / "anchors",
                chunk_index=chunk_index,
                save_anchor_debug=args.save_anchor_debug,
            )
            tracker_state = create_tracker_state(tracker, staged_dir, args)
            reacquired = find_reacquisition(
                frame_paths=selected_frame_paths,
                chunk_start=lost_frame + 1,
                chunk_end=chunk_end,
                anchor_bank=anchor_bank,
                args=args,
                events_path=events_path,
                chunk_index=chunk_index,
                global_start_frame=args.start_frame,
            )
            if reacquired is None:
                append_jsonl(
                    events_path,
                    {
                        "event": "within_chunk_reacquire_failed",
                        "chunk_index": chunk_index,
                        "frame_idx": args.start_frame + lost_frame,
                    },
                )
                save_untracked_frames(
                    frame_paths=selected_frame_paths,
                    start_idx=lost_frame + 1,
                    end_idx=chunk_end,
                    masks_dir=session_dir / "masks",
                    events_path=events_path,
                    global_start_frame=args.start_frame,
                )
                propagate_start = None
                break
            reacquire_frame_idx, reacquire_box, reacquire_score = reacquired
            seed_with_box(
                tracker=tracker,
                tracker_state=tracker_state,
                frame_idx=reacquire_frame_idx,
                obj_id=args.obj_id,
                box_xyxy=reacquire_box,
                frame_path=selected_frame_paths[reacquire_frame_idx],
                image_size=args.image_size,
            )
            prompt_box_by_frame = {reacquire_frame_idx: (reacquire_box, REACQUIRE_BOX_COLOR)}
            summary["within_chunk_recoveries"].append(
                {
                    "chunk_index": chunk_index,
                    "lost_frame_idx": args.start_frame + lost_frame,
                    "reacquired_frame_idx": args.start_frame + reacquire_frame_idx,
                    "score": reacquire_score,
                    "box_xyxy": reacquire_box,
                }
            )
            append_jsonl(
                events_path,
                {
                    "event": "within_chunk_reacquired",
                    "chunk_index": chunk_index,
                    "lost_frame_idx": args.start_frame + lost_frame,
                    "reacquired_frame_idx": args.start_frame + reacquire_frame_idx,
                    "score": reacquire_score,
                    "box_xyxy": reacquire_box,
                },
            )
            if reacquire_frame_idx > lost_frame + 1:
                save_untracked_frames(
                    frame_paths=selected_frame_paths,
                    start_idx=lost_frame + 1,
                    end_idx=reacquire_frame_idx - 1,
                    masks_dir=session_dir / "masks",
                    events_path=events_path,
                    global_start_frame=args.start_frame,
                )
            propagate_start = reacquire_frame_idx
            empty_cuda_cache()

        anchor_bank = build_anchor_bank(
            frame_records=frame_records,
            frame_paths=selected_frame_paths,
            max_anchors=args.max_anchors,
        )
        if not anchor_bank and chunk_index == 0:
            raise RuntimeError(
                "No usable anchors were produced in the first chunk. Check the seed box and checkpoint."
            )
        if anchor_bank:
            anchor_metadata = persist_anchor_bank(
                anchor_bank=anchor_bank,
                anchor_dir=session_dir / "anchors",
                chunk_index=chunk_index,
                save_anchor_debug=args.save_anchor_debug,
            )
            summary["anchor_bank"] = anchor_metadata
        append_jsonl(
            events_path,
            {
                "event": "chunk_end",
                "chunk_index": chunk_index,
                "chunk_start": chunk_global_start,
                "chunk_end": chunk_global_end,
                "anchor_frames": [args.start_frame + anchor.frame_idx for anchor in anchor_bank],
            },
        )
        del tracker_state
        empty_cuda_cache()

    summary["frames_tracked"] = sorted(set(summary["frames_tracked"]))
    summary["frames_lost"] = sorted(set(summary["frames_lost"]))
    write_json(session_dir / "summary.json", summary)
    print(f"Session written to: {session_dir}")


if __name__ == "__main__":
    main()
