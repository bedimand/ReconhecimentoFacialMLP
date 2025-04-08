import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import time
import os
from insightface.app import FaceAnalysis

# Import the FaceMLP model from predictor.py
from predictor import FaceMLP

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

def detect_and_analyze_faces(frame, face_analyzer):
    """
    Detect faces and extract facial landmarks using InsightFace
    
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
    Prepare a detected face for recognition using InsightFace detection
    
    Args:
        frame: Input frame
        face: InsightFace face object
        
    Returns:
        Processed face image as PIL Image
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
    
    # Resize to 92x112 (AT&T Database dimensions)
    face_resized = cv2.resize(face_gray, (92, 112))
    
    # Convert to PIL Image
    pil_img = Image.fromarray(face_resized)
    
    return pil_img

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
        # Use 5-point landmarks - this is the method that's working well
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
            print(f"5-point check - Horiz ratio: {horiz_ratio:.2f}, Vert ratio: {vert_ratio:.2f} (dev: {vert_deviation:.2f}) -> {'Looking' if is_looking else 'Not looking'}")
            return is_looking
        
        return True  # Default to looking if kps not available
    
    except Exception as e:
        print(f"Error in looking detection: {e}")
        return True  # Default to looking if there's an error

def main():
    # Configuration
    model_path = "face_mlp.pth"
    confidence_threshold = 80.0
    
    # Setup device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load classes from model file
    try:
        # Load the model checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Check if the model file contains class information
        if isinstance(checkpoint, dict) and 'classes' in checkpoint:
            classes = checkpoint['classes']
            model_state = checkpoint['model_state_dict']
            print(f"Loaded {len(classes)} classes from model file: {classes}")
        else:
            # Fallback to scanning dataset directory
            print("Model doesn't contain class info. Scanning dataset directory...")
            dataset_path = "dataset/"
            classes = []
            if os.path.exists(dataset_path):
                for item in os.listdir(dataset_path):
                    item_path = os.path.join(dataset_path, item)
                    if os.path.isdir(item_path):
                        classes.append(item)
            classes.sort()
            model_state = checkpoint
            print(f"Found {len(classes)} classes: {classes}")
        
        # Initialize model
        num_classes = len(classes)
        input_size = 92 * 112 * 1  # grayscale 92x112
        model = FaceMLP(input_size, num_classes)
        
        # Load model weights
        model.load_state_dict(model_state)
        model.to(device)
        model.eval()
        print(f"Face recognition model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Initialize face analyzer with InsightFace
    face_analyzer = initialize_face_analyzer()
    if face_analyzer is None:
        return
    
    # Preprocessing transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set frame size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Real-time face recognition started. Press 'q' to quit.")
    
    # Main loop
    detection_fps = 0  # For smoother FPS display
    show_landmarks = True  # Toggle for showing landmarks
    check_looking = True  # Toggle for checking if person is looking at camera
    looking_threshold = 0.75  # Threshold for the looking detection (0.0-1.0)
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Start timing for FPS calculation
        start_time = time.time()
        
        # Make a copy for display
        display_frame = frame.copy()
        
        # Detect and analyze faces with InsightFace
        faces = detect_and_analyze_faces(frame, face_analyzer)
        
        # Process each detected face
        for face in faces:
            # Get bounding box
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            
            # Check if person is looking at the camera
            looking_at_camera = not check_looking or is_looking_at_camera(face, looking_threshold)
            
            # Only process face recognition if looking at camera or check is disabled
            if looking_at_camera:
                # Prepare face for recognition
                face_img = prepare_face_for_recognition(frame, face)
                
                # Convert to tensor
                face_tensor = transform(face_img).unsqueeze(0).to(device)
                
                # Recognize face
                with torch.no_grad():
                    outputs = model(face_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                    
                # Get predicted class and confidence
                predicted_idx = outputs.argmax(dim=1).item()
                confidence = probabilities[predicted_idx].item() * 100
                predicted_label = classes[predicted_idx]
                
                # Draw bounding box - green for high confidence, orange for low
                color = (0, 255, 0) if confidence >= confidence_threshold else (0, 165, 255)
                
                # Display name and confidence
                if confidence >= confidence_threshold:
                    label = f"{predicted_label}: {confidence:.1f}%"
                else:
                    label = f"Unknown: {confidence:.1f}%"
            else:
                # If not looking at camera, display message and use different color
                color = (0, 0, 255)  # Red for not looking
                label = "Look at the camera"
            
            # Always draw the bounding box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # Always display the label
            cv2.putText(display_frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw facial landmarks if enabled
            if show_landmarks:
                draw_landmarks(display_frame, face, color=color)
        
        # Calculate processing time and FPS
        process_time = (time.time() - start_time) * 1000  # ms
        
        # Smooth FPS calculation for display
        current_fps = 1.0 / (time.time() - start_time)
        detection_fps = 0.9 * detection_fps + 0.1 * current_fps if detection_fps > 0 else current_fps
        
        # Display FPS and face count
        cv2.putText(display_frame, f"FPS: {detection_fps:.1f}, Processing: {process_time:.1f}ms", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.putText(display_frame, f"Faces: {len(faces)} | L: landmarks | C: looking", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display the resulting frame
        cv2.imshow("Face Recognition", display_frame)
        
        # Check for key presses
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('l'):
            # Toggle landmarks display
            show_landmarks = not show_landmarks
            print(f"Landmarks display: {'ON' if show_landmarks else 'OFF'}")
        elif key == ord('c'):
            # Toggle looking at camera check
            check_looking = not check_looking
            print(f"Looking at camera check: {'ON' if check_looking else 'OFF'}")
        elif key == ord('+') or key == ord('='):
            # Increase threshold
            looking_threshold = min(looking_threshold + 0.05, 1.0)
            print(f"Looking threshold increased to {looking_threshold:.2f}")
        elif key == ord('-'):
            # Decrease threshold
            looking_threshold = max(looking_threshold - 0.05, 0.0)
            print(f"Looking threshold decreased to {looking_threshold:.2f}")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 