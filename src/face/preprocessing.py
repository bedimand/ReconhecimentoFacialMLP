import os
# Suppress TensorFlow C++ and Abseil logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except ImportError:
    pass

import cv2
import numpy as np
import mediapipe as mp
from src.utils.config import config
from src.face.face_recognition import TransformFactory

class PreprocessModule:
    def __init__(self, target_size=None):
        """
        Initialize the PreprocessModule for background removal, resizing, and normalization.

        Args:
        - target_size (tuple): The target size for the resized image, defaults to value from config.
        """
        # Get target size from config if not provided
        if target_size is None:
            self.target_size = tuple(config.get('image_processing.target_size', [128, 128]))
        else:
            self.target_size = target_size
        
        # Initialize MediaPipe Selfie Segmentation for background removal
        self.mp_seg = mp.solutions.selfie_segmentation
        self.segmenter = self.mp_seg.SelfieSegmentation(model_selection=1)

    def remove_background_grabcut(self, face_image):
        """
        Remove background from the face image using MediaPipe Selfie Segmentation

        Args:
        - face_image (numpy.ndarray): The cropped face image (BGR format).

        Returns:
        - face_with_no_bg (numpy.ndarray): The face with the background removed.
        """
        # Convert BGR to RGB for MediaPipe
        rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        results = self.segmenter.process(rgb)
        if results.segmentation_mask is None:
            return face_image
        # Create binary mask: 1 for foreground, 0 for background
        mask = (results.segmentation_mask > 0.5).astype(np.uint8)
        # Expand mask to 3 channels
        mask_3c = np.stack([mask] * 3, axis=-1)
        # Apply mask to keep only foreground
        face_with_no_bg = face_image * mask_3c
        return face_with_no_bg

    def resize_image(self, face_image):
        """
        Resize the image to the target size.

        Args:
        - face_image (numpy.ndarray): The image to be resized.

        Returns:
        - face_resized (numpy.ndarray): The resized image.
        """
        return cv2.resize(face_image, self.target_size)

    def preprocess_face(self, face_image):
        """
        Complete preprocessing pipeline for the cropped face image:
        - Remove background
        - Return tensor processed via TransformFactory for consistent normalization

        Args:
        - face_image (numpy.ndarray): The cropped face image (BGR format).

        Returns:
        - processed_face (torch.Tensor): The processed face image tensor.
        """
        # Background removal
        face_with_no_bg = self.remove_background_grabcut(face_image)
        # Use TransformFactory for consistent normalization with rest of application
        processed_face = TransformFactory.preprocess_face_tensor(face_with_no_bg)
        return processed_face
