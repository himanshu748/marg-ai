import cv2
import numpy as np

from .config import VisionConfig


def bbox_area_m2(
    bbox: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    config: VisionConfig,
) -> float:
    src = np.float32(
        [
            [config.road_plane[3][0] * image_width, config.road_plane[3][1] * image_height],
            [config.road_plane[2][0] * image_width, config.road_plane[2][1] * image_height],
            [config.road_plane[1][0] * image_width, config.road_plane[1][1] * image_height],
            [config.road_plane[0][0] * image_width, config.road_plane[0][1] * image_height],
        ]
    )
    dst = np.float32([[0.0, 0.0], [3.5, 0.0], [3.5, 12.0], [0.0, 12.0]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    x, y, width, height = bbox
    corners = np.float32([[[x, y], [x + width, y], [x + width, y + height], [x, y + height]]])
    warped = cv2.perspectiveTransform(corners, matrix)[0]
    return float(abs(cv2.contourArea(warped)))


def score(class_name: str, area_m2: float) -> int:
    if class_name == "D40":
        return 5 if area_m2 > 0.5 else 4 if area_m2 > 0.2 else 3
    if class_name == "D20":
        return 4 if area_m2 > 2.0 else 3
    if class_name in {"D00", "D10"}:
        return 1 if area_m2 < 0.1 else 2
    return 1
