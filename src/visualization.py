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
