import cv2
import numpy as np
from typing import Tuple, Optional, Union, Callable

from plugins.cv2_visualization_plugin.gui_component import GuiComponent
from plugins.yolo_pose_plugin.data_structure import PoseData, COCO_CONNECTIONS

class GUIPose(GuiComponent):
    """A component to display pose estimation skeletons."""
    def __init__(
        self, 
        name: str, 
        parent: Optional[GuiComponent] = None, 
        position: Tuple[int, int] = (0, 0), 
        width: Union[int, Callable[[int], int], str] = 640, 
        height: Union[int, Callable[[int], int], str] = 480
    ):
        """Initializes the component."""
        super().__init__(name, width, height, parent, position)
        self.pose_data: PoseData | None = None

    def set_pose_data(self, pose_data: PoseData):
        """Sets the pose data to be rendered."""
        self.pose_data = pose_data

    def draw(self):
        """Draws the skeletons on the canvas."""
        # Create a transparent canvas
        self.canvas = np.zeros((self.height, self.width, 4), dtype=np.uint8)

        if self.pose_data:
            for skeleton in self.pose_data.skeletons:
                # Draw joints
                for joint in skeleton.joints:
                    if joint.confidence > 0.5: # Draw only confident joints
                        x = int(joint.x * self.width)
                        y = int(joint.y * self.height)
                        cv2.circle(self.canvas, (x, y), 5, (0, 255, 0, 255), -1)

                # Draw skeleton lines using connections from the COCO dataset
                for i, j in COCO_CONNECTIONS:
                    if i < len(skeleton.joints) and j < len(skeleton.joints):
                        joint1 = skeleton.joints[i]
                        joint2 = skeleton.joints[j]
                        if joint1.confidence > 0.5 and joint2.confidence > 0.5:
                            pt1 = (int(joint1.x * self.width), int(joint1.y * self.height))
                            pt2 = (int(joint2.x * self.width), int(joint2.y * self.height))
                            cv2.line(self.canvas, pt1, pt2, (255, 0, 0, 255), 2)
