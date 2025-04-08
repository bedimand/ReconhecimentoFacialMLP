import cv2
import numpy as np
import os
import time
from datetime import datetime
from insightface.app import FaceAnalysis

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
        print("Loaded modules: detection and facial landmarks only")
        return app
    except Exception as e:
        print(f"Error initializing InsightFace: {e}")
        print("Make sure you have installed insightface and onnxruntime packages:")
        print("pip install insightface onnxruntime")
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

def draw_landmarks(frame, face, color=(0, 255, 255)):
    """
    Draw facial landmarks on the frame using InsightFace landmarks
    
    Args:
        frame: Frame to draw on
        face: InsightFace face object with landmarks
        color: Color of the landmarks
    """
    # Get the 106 facial landmarks (kps is 5 points, landmark_3d_106 or landmark_2d_106 is 106 points)
    if face.landmark_2d_106 is not None:
        landmarks = face.landmark_2d_106
        
        # Draw all landmarks
        for point in landmarks.astype(np.int32):
            cv2.circle(frame, tuple(point), 1, color, -1)
        
        # Define facial features groups (approximate indices for 106-point model)
        facial_features = [
            ("Jaw", range(0, 33)),              # Jaw and cheeks
            ("Eyebrows", range(33, 51)),        # Left and right eyebrows 
            ("Nose", range(51, 60)),            # Nose bridge to nose tip
            ("Eyes", range(60, 76)),            # Left and right eyes
            ("Mouth", range(76, 106))           # Outer and inner lips
        ]
        
        # Draw lines connecting landmarks for each feature
        for name, feature_range in facial_features:
            points = landmarks[list(feature_range)].astype(np.int32)
            for i in range(len(points) - 1):
                cv2.line(frame, tuple(points[i]), tuple(points[i + 1]), color, 1)
            
            # Close the loop for eyes and mouth
            if name in ["Eyes", "Mouth"]:
                mid_idx = len(points) // 2
                # Left eye/mouth part
                cv2.line(frame, tuple(points[0]), tuple(points[mid_idx-1]), color, 1)
                # Right eye/mouth part
                cv2.line(frame, tuple(points[mid_idx]), tuple(points[-1]), color, 1)

def prepare_face_for_recognition(frame, face):
    """
    Prepare a detected face for saving using InsightFace detection with enhanced preprocessing
    
    Args:
        frame: Input frame
        face: InsightFace face object
        
    Returns:
        Processed face image ready for saving
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
    
    # Convert to grayscale
    face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    # 1. Face Alignment using key points
    if hasattr(face, 'kps') and face.kps is not None:
        kps = face.kps  # 5 key points: [left_eye, right_eye, nose, left_mouth, right_mouth]
        
        # Calculate eye center and angle
        left_eye = kps[0]
        right_eye = kps[1]
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

def save_face(frame, face, output_dir, face_count):
    """
    Process and save a detected face using InsightFace
    
    Args:
        frame: Source frame
        face: InsightFace face object
        output_dir: Directory to save the face
        face_count: Current face counter
        
    Returns:
        Path to saved face image
    """
    # Prepare face for saving
    face_img_resized = prepare_face_for_recognition(frame, face)
    
    # Save cropped face
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/face_{timestamp}_{face_count}.pgm"
    cv2.imwrite(filename, face_img_resized)
    
    return filename

def main():
    # Create output directory
    output_dir = "detected_faces"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize face analyzer with InsightFace
    face_analyzer = initialize_face_analyzer()
    if face_analyzer is None:
        return
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    # Check if webcam is opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set frame size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Face detection started. Press 'q' to quit. Press 's' to save current faces.")
    
    # Face detection loop
    face_count = 0
    detection_fps = 0  # Track detection FPS
    show_landmarks = True  # Toggle to show facial landmarks
    auto_save = False  # Toggle automatic saving
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Process frame for face detection
        start_time = time.time()
        
        # Detect faces using InsightFace
        faces = detect_faces(frame, face_analyzer)
        
        # Create display frame with boxes and info
        display_frame = frame.copy()
        
        # Process each detected face
        for i, face in enumerate(faces):
            # Get bounding box
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            
            # Draw bounding box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw facial landmarks if enabled
            if show_landmarks:
                draw_landmarks(display_frame, face, color=(0, 255, 255))
            
            # Save face if auto-save is enabled
            if auto_save:
                filename = save_face(frame, face, output_dir, face_count)
                print(f"Face saved: {filename}")
                face_count += 1
        
        # Calculate processing time and FPS
        detection_time = (time.time() - start_time) * 1000  # ms
        detection_fps = 0.9 * detection_fps + 0.1 * (1.0 / (time.time() - start_time)) if detection_fps > 0 else 1.0 / (time.time() - start_time)
        
        # Add text information to the display frame
        cv2.putText(display_frame, f"FPS: {detection_fps:.1f}, Detection: {detection_time:.1f}ms", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.putText(display_frame, f"Faces: {len(faces)}, Total Saved: {face_count}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        auto_save_status = "ON" if auto_save else "OFF"
        cv2.putText(display_frame, f"Auto-save: {auto_save_status} | L: landmarks | A: auto-save", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display the result
        cv2.imshow("Face Detection", display_frame)
        
        # Check for key presses
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save all current faces
            for face in faces:
                filename = save_face(frame, face, output_dir, face_count)
                print(f"Face saved: {filename}")
                face_count += 1
        elif key == ord('l'):
            # Toggle landmarks display
            show_landmarks = not show_landmarks
            print(f"Landmarks display: {'ON' if show_landmarks else 'OFF'}")
        elif key == ord('a'):
            # Toggle auto-save
            auto_save = not auto_save
            print(f"Auto-save: {'ON' if auto_save else 'OFF'}")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print(f"Total faces saved: {face_count}")

if __name__ == "__main__":
    main() 