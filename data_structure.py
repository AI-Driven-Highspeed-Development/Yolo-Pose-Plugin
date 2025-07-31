from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Joint:
    """
    Represents a single keypoint or joint detected by the pose estimation model.
    """
    x: float
    y: float
    confidence: float
    label: str

@dataclass
class Skeleton:
    """
    Represents the full set of joints for a single detected person.
    """
    joints: List[Joint]
    confidence: float
    bounding_box: Tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max)

@dataclass
class PoseData:
    """
    Container for all the skeletons detected in a single frame.
    """
    skeletons: List[Skeleton]

# COCO dataset keypoint connections
# See: https://cocodataset.org/#keypoints-eval
COCO_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # Body
    (5, 11), (6, 12), (11, 12), # Hips
    (11, 13), (13, 15), (12, 14), (14, 16) # Legs
]

# COCO dataset keypoint names
# See: https://cocodataset.org/#keypoints-eval
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]
