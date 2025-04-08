import cv2
import numpy as np

def preprocess_face(face_img, face_landmarks=None):
    """
    Apply preprocessing to face images
    
    Args:
        face_img: Face image (BGR format from OpenCV)
        face_landmarks: Optional facial landmarks for alignment
        
    Returns:
        Processed face image
    """
    # Convert to grayscale if not already
    if len(face_img.shape) == 3:
        face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        face_gray = face_img.copy()
    
    # 1. Face Alignment if landmarks are provided
    if face_landmarks is not None:
        # Extract eye landmarks
        left_eye = face_landmarks[0]
        right_eye = face_landmarks[1]
        
        # Calculate angle for alignment
        eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
        angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
        
        # Get rotation matrix
        height, width = face_gray.shape
        center = (width / 2, height / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Perform rotation
        face_gray = cv2.warpAffine(face_gray, rotation_matrix, (width, height), 
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    # 2. Histogram Equalization
    face_gray = cv2.equalizeHist(face_gray)
    
    # 3. Edge Enhancement
    # Apply Sobel operators
    sobelx = cv2.Sobel(face_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(face_gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Combine edges
    edges = np.sqrt(sobelx**2 + sobely**2)
    edges = np.uint8(edges)
    
    # Enhance original image with edges
    alpha = 0.3  # Edge enhancement factor
    face_gray = cv2.addWeighted(face_gray, 1.0, edges, alpha, 0)
    
    # Resize to 92x112 (AT&T Database dimensions)
    face_resized = cv2.resize(face_gray, (92, 112))
    
    return face_resized 