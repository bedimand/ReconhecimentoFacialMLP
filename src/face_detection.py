import cv2
import numpy as np
import os
from datetime import datetime
from insightface.app import FaceAnalysis
from src.preprocessing import preprocess_face

def initialize_face_analyzer():
    """
    Initialize the InsightFace analyzer for face detection and landmark detection
    """
    # Check if model directory exists
    model_dir = "models/insightface"
    if not os.path.exists(model_dir):
        print(f"InsightFace model directory not found at {model_dir}")
        print("Please run download_models.py first.")
        return None
    
    try:
        # Initialize the FaceAnalysis application with specific configs
        app = FaceAnalysis(name="buffalo_s", root=os.path.dirname(model_dir), 
                          allowed_modules=['detection', 'landmark_2d_106'])
        app.prepare(ctx_id=0, det_size=(640, 640))  # Use GPU if available (ctx_id=0)
        print("InsightFace model loaded successfully!")
        return app
    except Exception as e:
        print(f"Error initializing InsightFace: {e}")
        print("Make sure you have installed insightface and onnxruntime packages")
        return None

def detect_faces(frame, face_analyzer):
    """
    Detect faces using InsightFace
    
    Args:
        frame: Input frame
        face_analyzer: InsightFace FaceAnalysis object
        
    Returns:
        List of face objects containing detection and landmark data
    """
    if face_analyzer is None:
        return []
    
    # Detect faces (returns a list of Face objects with bbox, kps, landmark etc.)
    faces = face_analyzer.get(frame)
    
    return faces

def is_looking_at_camera(face, threshold=0.8):
    """
    Determine if the person is looking at the camera based on facial landmarks
    
    Args:
        face: InsightFace face object with landmarks
        threshold: Threshold for the looking score (0.0-1.0)
        
    Returns:
        Boolean indicating if the person is looking at the camera
    """
    try:
        # Use 5-point landmarks
        if hasattr(face, 'kps') and face.kps is not None:
            kps = face.kps  # 5 key points: [left_eye, right_eye, nose, left_mouth, right_mouth]
            
            # Extract landmark points
            left_eye = kps[0]
            right_eye = kps[1]
            nose = kps[2]
            
            # Check horizontal alignment (left-right)
            eye_center_x = (left_eye[0] + right_eye[0]) / 2
            nose_offset_x = abs(nose[0] - eye_center_x)
            eye_distance = abs(right_eye[0] - left_eye[0])
            horiz_ratio = nose_offset_x / eye_distance if eye_distance > 0 else 0
            
            # Check vertical alignment (up-down)
            eye_y = (left_eye[1] + right_eye[1]) / 2  # Average eye height
            eye_nose_distance = abs(nose[1] - eye_y)  # Vertical distance from eyes to nose
            vert_ratio = eye_nose_distance / eye_distance if eye_distance > 0 else 0
            
            # Typical ratio is around 0.4-0.6 for frontal face
            normal_vert_ratio = 0.5  # Approximate normal ratio
            vert_deviation = abs(vert_ratio - normal_vert_ratio)
            
            # Thresholds for deciding if looking
            horiz_ok = horiz_ratio < 0.15  # Nose should be within 15% of eye center
            vert_ok = vert_deviation < 0.2  # Vertical ratio should be within 20% of normal
            
            is_looking = horiz_ok and vert_ok
            return is_looking
        
        return True  # Default to looking if kps not available
    
    except Exception as e:
        print(f"Error in looking detection: {e}")
        return True  # Default to looking if there's an error

def extract_face(frame, face):
    """
    Extract face from frame with preprocessing
    
    Args:
        frame: Input frame
        face: InsightFace face object
        
    Returns:
        Processed face image
    """
    # Get bounding box coordinates
    bbox = face.bbox.astype(int)
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    
    # Add margin and ensure within frame bounds
    frame_h, frame_w = frame.shape[:2]
    margin = 20
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(frame_w, x2 + margin)
    y2 = min(frame_h, y2 + margin)
    
    # Crop face
    face_img = frame[y1:y2, x1:x2]
    
    # Apply preprocessing with landmarks if available
    if hasattr(face, 'kps') and face.kps is not None:
        # Adjust the landmark coordinates relative to the cropped region
        landmarks = face.kps.copy()
        landmarks[:, 0] -= x1
        landmarks[:, 1] -= y1
        
        # Process face
        processed_face = preprocess_face(face_img, landmarks)
    else:
        # Process without alignment
        processed_face = preprocess_face(face_img)
    
    return processed_face

def save_face(frame, face, output_dir, person_name=None):
    """
    Process and save a detected face
    
    Args:
        frame: Source frame
        face: InsightFace face object
        output_dir: Base directory to save faces
        person_name: Optional name of the person (for creating subdirectory)
        
    Returns:
        Path to saved face image
    """
    # Create output directory
    if person_name:
        # Create a subfolder for this person
        person_dir = os.path.join(output_dir, person_name)
        os.makedirs(person_dir, exist_ok=True)
        save_dir = person_dir
    else:
        # Save to the main output directory
        os.makedirs(output_dir, exist_ok=True)
        save_dir = output_dir
    
    # Process face
    face_img = extract_face(frame, face)
    
    # Save face image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{save_dir}/face_{timestamp}.pgm"
    cv2.imwrite(filename, face_img)
    
    return filename
