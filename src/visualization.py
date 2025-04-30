import cv2
import numpy as np

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

def draw_five_landmarks(frame, face, show_labels=False, point_size=5, line_thickness=2):
    """
    Draw the 5 key facial landmarks with connections
    
    Args:
        frame: Input frame to draw on
        face: InsightFace face object with landmarks
        show_labels: Whether to show labels for each point
        point_size: Size of the points to draw
        line_thickness: Thickness of the connection lines
        
    Returns:
        Frame with landmarks drawn
    """
    if not hasattr(face, 'kps') or face.kps is None:
        return frame
    
    # Make a copy of the frame
    vis_frame = frame.copy()
    
    # Get 5 facial landmarks
    kps = face.kps  # [left_eye, right_eye, nose, left_mouth, right_mouth]
    
    # Point colors - BGR format
    landmark_colors = [
        (0, 255, 0),    # Left eye - Green
        (0, 255, 0),    # Right eye - Green
        (0, 0, 255),    # Nose - Red
        (255, 0, 0),    # Left mouth - Blue
        (255, 0, 0)     # Right mouth - Blue
    ]
    
    # Point labels
    landmark_labels = ["L-Eye", "R-Eye", "Nose", "L-Mouth", "R-Mouth"]
    
    # Draw each landmark
    for i, (point, color) in enumerate(zip(kps, landmark_colors)):
        x, y = int(point[0]), int(point[1])
        # Draw circle
        cv2.circle(vis_frame, (x, y), point_size, color, -1)
        
        # Draw label if requested
        if show_labels:
            cv2.putText(vis_frame, landmark_labels[i], (x + 5, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Draw connections between landmarks to show face structure
    # Eyes to nose
    cv2.line(vis_frame, (int(kps[0][0]), int(kps[0][1])), 
             (int(kps[2][0]), int(kps[2][1])), (255, 255, 0), line_thickness)
    cv2.line(vis_frame, (int(kps[1][0]), int(kps[1][1])), 
             (int(kps[2][0]), int(kps[2][1])), (255, 255, 0), line_thickness)
    
    # Nose to mouth
    cv2.line(vis_frame, (int(kps[2][0]), int(kps[2][1])), 
             (int(kps[3][0]), int(kps[3][1])), (255, 255, 0), line_thickness)
    cv2.line(vis_frame, (int(kps[2][0]), int(kps[2][1])), 
             (int(kps[4][0]), int(kps[4][1])), (255, 255, 0), line_thickness)
    
    # Mouth connection
    cv2.line(vis_frame, (int(kps[3][0]), int(kps[3][1])), 
             (int(kps[4][0]), int(kps[4][1])), (255, 255, 0), line_thickness)
    
    return vis_frame

def draw_nose_centered_crop(frame, face, crop_size=(128, 128)):
    """
    Visualize how the nose-centered crop will be performed showing a tighter face crop
    
    Args:
        frame: Input frame
        face: InsightFace face object with landmarks
        crop_size: Target crop size (width, height)
        
    Returns:
        Frame with crop visualization drawn
    """
    if not hasattr(face, 'kps') or face.kps is None:
        return frame
    
    # Make a copy of the frame
    vis_frame = frame.copy()
    
    # Get frame dimensions
    frame_h, frame_w = vis_frame.shape[:2]
    
    # Get face bounding box
    bbox = face.bbox.astype(int)
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    
    # Calculate face dimensions
    face_width = x2 - x1
    face_height = y2 - y1
    
    # Get nose position
    nose = face.kps[2]  # Nose is the third keypoint
    nose_x, nose_y = int(nose[0]), int(nose[1])
    
    # Get eye positions to better estimate face size
    left_eye = face.kps[0]
    right_eye = face.kps[1]
    eye_distance = np.sqrt((left_eye[0] - right_eye[0])**2 + (left_eye[1] - right_eye[1])**2)
    
    # Use eye distance to determine crop size (typically faces are ~2.5-3x eye distance in width)
    face_scale = 2.2  # This controls how tight the crop will be (lower = tighter)
    ideal_size = int(eye_distance * face_scale)
    
    # Calculate square region around nose using the ideal size
    half_size = ideal_size // 2
    square_x1 = max(0, nose_x - half_size)
    square_y1 = max(0, nose_y - half_size)
    square_x2 = min(frame_w, nose_x + half_size)
    square_y2 = min(frame_h, nose_y + half_size)
    
    # Check if we hit the image boundaries and adjust
    if square_x1 == 0:
        square_x2 = min(frame_w, square_x1 + ideal_size)
    if square_y1 == 0:
        square_y2 = min(frame_h, square_y1 + ideal_size)
    if square_x2 == frame_w:
        square_x1 = max(0, square_x2 - ideal_size)
    if square_y2 == frame_h:
        square_y1 = max(0, square_y2 - ideal_size)
    
    # Draw the 5 landmarks
    vis_frame = draw_five_landmarks(vis_frame, face, show_labels=True)
    
    # Draw a circle around the nose to show the center point
    cv2.circle(vis_frame, (nose_x, nose_y), 10, (0, 0, 255), 2)
    
    # Draw the crop box to show what will be extracted (square region)
    cv2.rectangle(vis_frame, (square_x1, square_y1), (square_x2, square_y2), (255, 255, 0), 2)
    
    # Draw the original face bounding box for reference
    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    
    # Draw eye distance line
    cv2.line(vis_frame, (int(left_eye[0]), int(left_eye[1])), 
             (int(right_eye[0]), int(right_eye[1])), (255, 0, 255), 2)
    
    # Draw crosshairs at the nose position
    line_length = 20
    cv2.line(vis_frame, (nose_x - line_length, nose_y), (nose_x + line_length, nose_y), (0, 255, 255), 1)
    cv2.line(vis_frame, (nose_x, nose_y - line_length), (nose_x, nose_y + line_length), (0, 255, 255), 1)
    
    # Add text labels
    cv2.putText(vis_frame, f"Nose-centered crop: {crop_size[0]}x{crop_size[1]}", 
               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(vis_frame, "Green: Face detection | Yellow: Nose-centered region", 
               (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(vis_frame, f"Eye distance: {eye_distance:.1f}px | Crop size: {ideal_size}px", 
               (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return vis_frame

def draw_face_info(frame, face, label, color=(0, 255, 0)):
    """
    Draw face bounding box and label
    
    Args:
        frame: Frame to draw on
        face: InsightFace face object
        label: Text label to display
        color: Color of box and text
    """
    # Get bounding box
    bbox = face.bbox.astype(int)
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    
    # Draw rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Draw label
    cv2.putText(frame, label, (x1, y1-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def draw_stats(frame, fps, process_time, face_count, extra_info=None):
    """
    Draw performance statistics and info on frame
    
    Args:
        frame: Frame to draw on
        fps: Current FPS
        process_time: Processing time in ms
        face_count: Number of detected faces
        extra_info: Additional info to display
    """
    # Draw FPS and processing time
    cv2.putText(frame, f"FPS: {fps:.1f}, Processing: {process_time:.1f}ms", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Draw face count
    cv2.putText(frame, f"Faces: {face_count}", 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Draw extra info if provided
    if extra_info:
        cv2.putText(frame, extra_info, 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

def draw_crosshair_center(image):
    """
    Draw crosshairs at the center of an image to show the exact center point
    
    Args:
        image: Image to draw on (will be modified in-place)
        
    Returns:
        Image with crosshairs
    """
    # Get image dimensions
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2
    
    # Draw crosshairs
    line_length = 10
    cv2.line(image, (center_x - line_length, center_y), 
             (center_x + line_length, center_y), (0, 255, 255), 1)
    cv2.line(image, (center_x, center_y - line_length), 
             (center_x, center_y + line_length), (0, 255, 255), 1)
    
    return image

def draw_recognition_results(frame, face, result):
    """
    Draw recognition results on frame
    
    Args:
        frame: Input camera frame
        face: Detected face information
        result: Recognition result dictionary from process_frame_for_recognition
        
    Returns:
        frame: Frame with annotations
    """
    # Get bounding box
    bbox = face.bbox.astype(int)
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), result['color'], 2)
    
    # Draw label
    text_position = (x1, y1 - 10)
    cv2.putText(
        frame, 
        result['label'], 
        text_position, 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.7,  # Scale
        result['color'], 
        2  # Thickness
    )
    
    # Draw 5-point landmarks if available and looking at camera
    if hasattr(face, 'kps') and face.kps is not None and result['looking_at_camera']:
        kps = face.kps
        # Draw small circles at each landmark
        for point in kps:
            cv2.circle(frame, (int(point[0]), int(point[1])), 2, (0, 255, 255), -1)
    
    return frame
