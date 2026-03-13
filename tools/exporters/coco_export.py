import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class ObjectInfo:
    obj_id: int
    name: str


@dataclass
class PointPrompt:
    x_px: int
    y_px: int
    is_positive: bool


@dataclass
class SamFrameOutput:
    obj_ids: List[int]
    masks: List[np.ndarray]
    boxes_xywh_norm: List[Tuple[float, float, float, float]]
    scores: List[float]


class CocoExporter:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.masks_dir = self.output_dir / "masks"
        self.masks_dir.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        frame_paths: List[Path],
        objects: List[ObjectInfo],
        prompts_by_frame_obj: Dict[int, Dict[int, List[PointPrompt]]],
        outputs_by_frame: Dict[int, SamFrameOutput],
        image_shape: Optional[Tuple[int, int]] = None,
        json_name: str = "annotations_coco.json",
    ) -> Path:
        categories = [
            {"id": obj.obj_id, "name": obj.name, "supercategory": "surgical_tool"}
            for obj in objects
        ]

        images = []
        annotations = []
        ann_id = 1

        for frame_idx, frame_path in enumerate(frame_paths):
            if image_shape is None:
                img = cv2.imread(str(frame_path))
                if img is None:
                    continue
                height, width = img.shape[:2]
            else:
                height, width = image_shape

            image_id = frame_idx + 1
            images.append(
                {
                    "id": image_id,
                    "file_name": frame_path.name,
                    "width": int(width),
                    "height": int(height),
                    "frame_index": frame_idx,
                }
            )

            frame_output = outputs_by_frame.get(frame_idx)
            if frame_output is None:
                continue

            obj_to_idx = {int(obj_id): i for i, obj_id in enumerate(frame_output.obj_ids)}
            for obj in objects:
                if obj.obj_id not in obj_to_idx:
                    continue

                idx = obj_to_idx[obj.obj_id]
                mask = frame_output.masks[idx].astype(np.uint8)
                if mask.max() == 0:
                    continue

                x_norm, y_norm, w_norm, h_norm = frame_output.boxes_xywh_norm[idx]
                x = float(x_norm * width)
                y = float(y_norm * height)
                w = float(w_norm * width)
                h = float(h_norm * height)

                area = int(mask.sum())
                mask_filename = f"frame_{frame_idx:05d}_obj_{obj.obj_id}.png"
                mask_path = self.masks_dir / mask_filename
                cv2.imwrite(str(mask_path), mask * 255)

                prompts = prompts_by_frame_obj.get(frame_idx, {}).get(obj.obj_id, [])
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": obj.obj_id,
                        "bbox": [x, y, w, h],
                        "area": area,
                        "iscrowd": 0,
                        "segmentation": [],
                        "mask_path": str(mask_path.relative_to(self.output_dir)),
                        "score": float(frame_output.scores[idx]),
                        "num_prompts": len(prompts),
                    }
                )
                ann_id += 1

        coco = {
            "info": {
                "description": "SAM3 Surgical Tool Annotations",
                "version": "1.0",
            },
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }

        out_json = self.output_dir / json_name
        out_json.write_text(json.dumps(coco, indent=2), encoding="utf-8")
        return out_json
