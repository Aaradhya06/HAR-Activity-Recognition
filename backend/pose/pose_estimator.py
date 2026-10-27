"""
MediaPipe Pose Estimation Service
===================================
Wraps MediaPipe Pose for real-time keypoint extraction.

Outputs 33 keypoints per frame, each with (x, y, z, visibility).
For the model we use (x, y, visibility) → 99-dim vector per frame.

Gracefully degrades if MediaPipe is not installed — returns zero arrays.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

NUM_KEYPOINTS = 33           # MediaPipe Pose landmark count
KEYPOINT_DIM = 3             # x, y, visibility per keypoint
POSE_VECTOR_DIM = NUM_KEYPOINTS * KEYPOINT_DIM  # 99


class PoseEstimator:
    """
    Wraps MediaPipe Pose for per-frame keypoint extraction.

    Args:
        min_detection_confidence: Minimum confidence for initial detection.
        min_tracking_confidence:  Minimum confidence for tracking.
        draw_landmarks: Whether to draw skeleton overlay on frames.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        draw_landmarks: bool = True,
    ) -> None:
        self.available = False
        self.draw_landmarks = draw_landmarks
        self._pose = None

        try:
            import mediapipe as mp
            self._mp = mp
            self._mp_pose = mp.solutions.pose
            self._mp_drawing = mp.solutions.drawing_utils
            self._mp_drawing_styles = mp.solutions.drawing_styles
            self._pose = self._mp_pose.Pose(
                static_image_mode=False,
                model_complexity=0,             # Lite model
                smooth_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self.available = True
            logger.info("MediaPipe Pose initialized (complexity=0, real-time mode)")
        except ImportError:
            logger.warning(
                "MediaPipe not installed. Pose estimation disabled. "
                "Install with: pip install mediapipe"
            )
        except Exception as e:
            logger.warning("Failed to initialize MediaPipe Pose: %s", e)

    def extract_keypoints(
        self,
        frame_bgr: np.ndarray,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Extract pose keypoints from a single BGR frame.

        Returns:
            (annotated_frame, keypoints_vector)
            - annotated_frame: BGR frame with optional skeleton drawn
            - keypoints_vector: (99,) float32 array, or None if detection failed
        """
        if not self.available or self._pose is None:
            return frame_bgr, np.zeros(POSE_VECTOR_DIM, dtype=np.float32)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self._pose.process(frame_rgb)
        frame_rgb.flags.writeable = True

        annotated = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks is None:
            return annotated, np.zeros(POSE_VECTOR_DIM, dtype=np.float32)

        # Extract keypoints
        kp_vector = np.array(
            [
                [lm.x, lm.y, lm.visibility]
                for lm in results.pose_landmarks.landmark
            ],
            dtype=np.float32,
        ).flatten()  # (99,)

        # Draw skeleton overlay
        if self.draw_landmarks:
            self._mp_drawing.draw_landmarks(
                annotated,
                results.pose_landmarks,
                self._mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self._mp_drawing_styles.get_default_pose_landmarks_style(),
            )

        return annotated, kp_vector

    def extract_sequence(
        self,
        frames_bgr: list[np.ndarray],
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """
        Process a sequence of frames.

        Returns:
            (annotated_frames, pose_sequence)
            - annotated_frames: list of BGR frames with skeleton drawn
            - pose_sequence: (T, 99) float32 array
        """
        annotated_frames = []
        keypoints_sequence = []

        for frame in frames_bgr:
            ann, kp = self.extract_keypoints(frame)
            annotated_frames.append(ann)
            keypoints_sequence.append(kp)

        pose_array = np.stack(keypoints_sequence, axis=0)  # (T, 99)
        return annotated_frames, pose_array

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._pose is not None:
            self._pose.close()
            self._pose = None

    def __del__(self) -> None:
        self.close()


# ── Singleton accessor ────────────────────────────────────────────────────────

_global_estimator: Optional[PoseEstimator] = None


def get_pose_estimator(
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    draw_landmarks: bool = True,
) -> PoseEstimator:
    """Get or create the global PoseEstimator singleton."""
    global _global_estimator
    if _global_estimator is None:
        _global_estimator = PoseEstimator(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            draw_landmarks=draw_landmarks,
        )
    return _global_estimator
