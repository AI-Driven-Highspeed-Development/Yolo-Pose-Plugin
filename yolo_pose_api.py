import numpy as np
from ultralytics import YOLO
from typing import Optional

from managers.config_manager.config_manager import ConfigManager
from .data_structure import PoseData, Skeleton, Joint, COCO_KEYPOINT_NAMES

class YoloPoseApi:
    """
    A simple API to interact with a YOLO pose estimation model.
    """
    def __init__(self):
        """
        Initializes the YoloPoseApi, loading the model and configuration.
        """
        self.config_manager = ConfigManager()
        
        # Get config with defaults from the ConfigManager
        plugin_config = getattr(self.config_manager.config, 'yolo_pose_plugin', None)
        
        self.model_name = getattr(plugin_config, 'model_name', 'yolov8n-pose.pt')
        self.confidence_threshold = float(getattr(plugin_config, 'confidence_threshold', 0.5))
        
        try:
            self.model = YOLO(self.model_name)
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            print("Please ensure the model name is correct and the file is accessible.")
            self.model = None

    def detect_poses(self, frame: np.ndarray) -> Optional[PoseData]:
        """
        Detects human poses in a given image frame.

        Args:
            frame (np.ndarray): The input image in BGR format.

        Returns:
            PoseData: An object containing all detected skeletons, or None if model failed to load.
        """
        if self.model is None:
            return None
            
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)

        skeletons = []
        for res in results:
            if res.keypoints is None or res.boxes is None:
                continue

            for i in range(len(res.boxes)):
                box = res.boxes[i].xyxyn[0].cpu().numpy()
                box_conf = float(res.boxes[i].conf[0].cpu().numpy())
                
                kpts = res.keypoints
                joints = []
                
                # Extract keypoints data using proper ultralytics API
                xy_data = kpts.xy[i].cpu().numpy()  # x,y coordinates
                conf_data = kpts.conf[i].cpu().numpy() if kpts.conf is not None else None  # confidence scores
                
                for kp_idx, (x, y) in enumerate(xy_data):
                    # Get confidence if available, otherwise use 1.0
                    conf = float(conf_data[kp_idx]) if conf_data is not None else 1.0
                    
                    # Create joint label based on COCO pose keypoint names
                    label = COCO_KEYPOINT_NAMES[kp_idx] if kp_idx < len(COCO_KEYPOINT_NAMES) else f'joint_{kp_idx}'
                    
                    # Normalize coordinates by image dimensions
                    img_height, img_width = frame.shape[:2]
                    x_norm = float(x) / img_width
                    y_norm = float(y) / img_height
                    
                    joints.append(Joint(x=x_norm, y=y_norm, confidence=conf, label=label))

                skeletons.append(Skeleton(
                    joints=joints,
                    confidence=box_conf,
                    bounding_box=(float(box[0]), float(box[1]), float(box[2]), float(box[3]))
                ))

        return PoseData(skeletons=skeletons)
