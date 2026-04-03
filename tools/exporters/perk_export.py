import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class ObjectInfo:
    obj_id: int
    name: str


@dataclass
class BoxPrompt:
    x1_px: int
    y1_px: int
    x2_px: int
    y2_px: int


class PerkExporter:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        frame_paths: List[Path],
        objects: List[ObjectInfo],
        boxes_by_frame_obj: Dict[int, Dict[int, BoxPrompt]],
        csv_name: str = "annotations_perk.csv",
    ) -> Path:
        object_names = {obj.obj_id: obj.name for obj in objects}
        out_csv = self.output_dir / csv_name
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["Filename", "Tool bounding box"])
            writer.writeheader()
            for frame_idx, frame_path in enumerate(frame_paths):
                per_obj = boxes_by_frame_obj.get(frame_idx, {})
                tool_boxes = []
                for obj_id in sorted(per_obj.keys()):
                    box = per_obj[obj_id]
                    tool_boxes.append(
                        {
                            "class": object_names.get(obj_id, str(obj_id)),
                            "xmin": int(box.x1_px),
                            "ymin": int(box.y1_px),
                            "xmax": int(box.x2_px),
                            "ymax": int(box.y2_px),
                        }
                    )
                writer.writerow(
                    {
                        "Filename": frame_path.name,
                        "Tool bounding box": repr(tool_boxes),
                    }
                )
        return out_csv
