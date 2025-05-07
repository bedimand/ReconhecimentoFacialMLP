from src.face.face_detection import extract_face, is_looking_at_camera
from src.face.face_recognition import recognize_face
from src.utils.config import config
from src.face.preprocessing import PreprocessModule

# Lazy-initialized background-removal preprocessor for real-time inference
_preprocessor = None

def process_frame_for_recognition(frame, face, model, classes, device, confidence_threshold=None, check_looking=None):
    """
    Process a detected face in a frame and apply face recognition
    
    Args:
        frame: Input camera frame
        face: Detected face from face detector
        model: Face recognition model
        classes: List of class names
        device: Torch device
        confidence_threshold: Threshold for recognition confidence (pulled from config if None)
        check_looking: Whether to check if person is looking at camera (pulled from config if None)
        
    Returns:
        result_dict: Dictionary containing recognition results
            - 'label': Display label with name and confidence
            - 'name': Recognized person name or 'unknown'
            - 'confidence': Recognition confidence score
            - 'color': BGR color tuple for display
            - 'looking_at_camera': Whether person is looking at camera
    """
    # Get parameters from config if not provided
    if confidence_threshold is None:
        confidence_threshold = config.get('recognition.confidence_threshold', 50.0)
    if check_looking is None:
        check_looking = config.get('recognition.looking_detection.enabled', True)
    
    # Get colors from config
    high_conf_color = config.get('visualization.bounding_box.high_confidence_color', [0, 255, 0])
    low_conf_color = config.get('visualization.bounding_box.low_confidence_color', [0, 165, 255])
    not_looking_color = config.get('visualization.bounding_box.not_looking_color', [0, 0, 255])
    
    # Initialize result dictionary
    result = {
        'confidence': 0.0,
        'name': 'unknown',
        'looking_at_camera': False
    }
    
    # Check if person is looking at camera
    looking_at_camera = not check_looking or is_looking_at_camera(face)
    result['looking_at_camera'] = looking_at_camera
    
    # Only process face recognition if looking at camera or check is disabled
    if looking_at_camera:
        # Extract face crop
        face_img = extract_face(frame, face)
        # Lazy initialize the background-removal preprocessor
        global _preprocessor
        if _preprocessor is None:
            try:
                _preprocessor = PreprocessModule()
            except Exception as e:
                print(f"Warning: could not initialize PreprocessModule: {e}")
        # Apply background removal if available
        if _preprocessor:
            face_img = _preprocessor.remove_background_grabcut(face_img)
        
        # Perform recognition
        predicted_class, confidence, all_probabilities = recognize_face(
            model, face_img, classes, device
        )
        
        # Store results
        result['confidence'] = confidence
        result['probabilities'] = all_probabilities
        
        # Determine if confidence is high enough
        if predicted_class.startswith('unknown') or confidence < confidence_threshold:
            result['name'] = 'unknown'
            result['color'] = tuple(low_conf_color)
            result['label'] = f"Desconhecido: {confidence:.1f}%"
        else:
            result['name'] = predicted_class
            result['color'] = tuple(high_conf_color)
            result['label'] = f"{predicted_class}: {confidence:.1f}%"
    else:
        # Not looking at camera
        result['color'] = tuple(not_looking_color)
        result['label'] = "Look at the camera"
    
    return result 