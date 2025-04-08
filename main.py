import os
import cv2
import torch
import numpy as np
import time
from PIL import Image
import threading
import queue

from src.config import config
from src.utils import (
    check_requirements, 
    get_device, 
    clear_console, 
    print_header,
    wait_key,
    create_person_dir,
    count_images_by_person
)
from src.face_detection import (
    initialize_face_analyzer, 
    detect_faces, 
    save_face
)
from src.face_recognition import (
    recognize_face, 
    evaluate_model,
    process_frame_for_recognition
)
from src.visualization import (
    draw_landmarks, 
    draw_face_info, 
    draw_stats
)
from src.training import train_model
from src.model import load_model

# Get paths from config
DATASET_PATH = config.get_dataset_dir()
MODEL_PATH = config.get_model_save_path()

# Function to detect and collect face images
def detect_and_collect_faces():
    # Create output directory
    os.makedirs(DATASET_PATH, exist_ok=True)
    
    # Initialize face analyzer
    face_analyzer = initialize_face_analyzer()
    if face_analyzer is None:
        print("Failed to initialize face detection. Please make sure models are downloaded.")
        wait_key()
        return
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        wait_key()
        return
    
    # Set frame size from config
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.get('camera.width'))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.get('camera.height'))
    
    # Ask for person's name
    print("Enter the person's name (or press Enter for unnamed faces):")
    person_name = input().strip()
    
    # Create person directory if name is provided
    if person_name:
        person_dir = create_person_dir(person_name, DATASET_PATH)
    else:
        person_dir = os.path.join(DATASET_PATH, "unknown")
        os.makedirs(person_dir, exist_ok=True)
    
    print_header("Face Detection and Collection")
    print(f"Press '{config.get('controls.save_face')}' to save current face")
    print(f"Press '{config.get('controls.toggle_auto_save')}' to toggle auto-save")
    print(f"Press '{config.get('controls.toggle_landmarks')}' to toggle landmarks")
    print(f"Press '{config.get('controls.quit')}' to quit")
    
    # Face detection loop
    face_count = 0
    detection_fps = 0
    show_landmarks = config.get('visualization.landmarks.enabled', True)
    auto_save = False
    
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
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), config.get_color('visualization.bounding_box.high_confidence_color'), 2)
            
            # Draw facial landmarks if enabled
            if show_landmarks:
                draw_landmarks(display_frame, face, color=config.get_color('visualization.landmarks.color'))
            
            # Save face if auto-save is enabled
            if auto_save:
                filename = save_face(frame, face, person_dir)
                print(f"Face saved: {filename}")
                face_count += 1
        
        # Calculate processing time and FPS
        detection_time = (time.time() - start_time) * 1000  # ms
        detection_fps = 0.9 * detection_fps + 0.1 * (1.0 / (time.time() - start_time)) if detection_fps > 0 else 1.0 / (time.time() - start_time)
        
        # Add text information to the display frame
        extra_info = f"Auto-save: {'ON' if auto_save else 'OFF'} | L: landmarks | A: auto-save"
        draw_stats(display_frame, detection_fps, detection_time, len(faces), extra_info)
        
        # Display the result
        cv2.imshow("Face Detection", display_frame)
        
        # Check for key presses
        key = cv2.waitKey(1)
        if key == ord(config.get('controls.quit')):
            break
        elif key == ord(config.get('controls.save_face')):
            # Save all current faces
            for face in faces:
                filename = save_face(frame, face, person_dir)
                print(f"Face saved: {filename}")
                face_count += 1
        elif key == ord(config.get('controls.toggle_landmarks')):
            # Toggle landmarks display
            show_landmarks = not show_landmarks
            print(f"Landmarks display: {'ON' if show_landmarks else 'OFF'}")
        elif key == ord(config.get('controls.toggle_auto_save')):
            # Toggle auto-save
            auto_save = not auto_save
            print(f"Auto-save: {'ON' if auto_save else 'OFF'}")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print(f"Total faces saved: {face_count}")
    wait_key()

# Function to train the recognition model
def train_recognition_model():
    print_header("Train Face Recognition Model")
    
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset path {DATASET_PATH} not found!")
        wait_key()
        return
    
    # Count images by person
    person_counts = count_images_by_person(DATASET_PATH)
    
    if not person_counts:
        print("No person folders found in dataset!")
        wait_key()
        return
    
    print("Available people in dataset:")
    for person, count in person_counts.items():
        print(f"- {person}: {count} images")
    
    # Get training parameters from config
    num_epochs = config.get('training.num_epochs')
    batch_size = config.get('training.batch_size')
    learning_rate = config.get('training.learning_rate')
    
    print("\nTraining parameters:")
    print(f"Number of epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    
    print("\nStarting training...")
    
    # Train the model
    model, classes = train_model(
        DATASET_PATH, 
        MODEL_PATH, 
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    if model is None:
        print("Training failed!")
    else:
        print(f"Model trained successfully and saved to {MODEL_PATH}")
    
    wait_key()
    return model, classes

# Function to evaluate the model
def evaluate_recognition_model():
    print_header("Evaluate Face Recognition Model")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file {MODEL_PATH} not found!")
        print("Please train the model first.")
        wait_key()
        return
    
    # Get device from config
    device = config.get_device()
    
    # Load model
    model, classes = load_model(MODEL_PATH, device)
    if model is None:
        print("Failed to load model!")
        wait_key()
        return
    
    print(f"Loaded model with {len(classes)} classes: {classes}")
    
    # Evaluate model
    accuracy = evaluate_model(model, DATASET_PATH, classes, device)
    
    print(f"\nOverall accuracy: {accuracy:.2f}%")
    wait_key()

# Function for real-time face recognition
def real_time_recognition(model=None, classes=None):
    print_header("Real-time Face Recognition")
    
    # Get device from config
    device = config.get_device()
    
    # Load model if not provided
    if model is None or classes is None:
        # Check if model exists
        if not os.path.exists(MODEL_PATH):
            print(f"Error: Model file {MODEL_PATH} not found!")
            print("Please train the model first.")
            wait_key()
            return
        
        # Load model
        model, classes = load_model(MODEL_PATH, device)
        if model is None:
            print("Failed to load model!")
            wait_key()
            return
    
    print(f"Loaded model with {len(classes)} classes: {classes}")
    
    # Initialize face analyzer
    face_analyzer = initialize_face_analyzer()
    if face_analyzer is None:
        print("Failed to initialize face detection. Please make sure models are downloaded.")
        wait_key()
        return
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        wait_key()
        return
    
    # Set frame size from config
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.get('camera.width'))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.get('camera.height'))
    
    print("Real-time face recognition started.")
    print(f"Press '{config.get('controls.toggle_landmarks')}' to toggle landmarks")
    print(f"Press '{config.get('controls.toggle_looking_detection')}' to toggle looking at camera check")
    print("Press 'a' to add a new person")
    print("Press 'r' to retrain model")
    print(f"Press '{config.get('controls.quit')}' to quit")
    
    # Main loop variables
    detection_fps = 0
    show_landmarks = config.get('visualization.landmarks.enabled', True)
    check_looking = config.get('recognition.looking_detection.enabled', True)
    confidence_threshold = config.get('recognition.confidence_threshold', 80.0)
    
    # Shared variables for capturing new person
    capturing_new_person = False
    new_person_name = ""
    
    # Function to handle model retraining in a thread
    def retrain_model_thread(q):
        print("Retraining model...")
        new_model, new_classes = train_model(
            DATASET_PATH, 
            MODEL_PATH, 
            num_epochs=config.get('training.num_epochs'),
            batch_size=config.get('training.batch_size'),
            learning_rate=config.get('training.learning_rate')
        )
        q.put((new_model, new_classes))
        print("Model retraining completed!")
    
    # Queue for thread communication
    model_queue = queue.Queue()
    retraining = False
    
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
        
        # Check if a new model is available from retraining
        if not model_queue.empty() and retraining:
            model, classes = model_queue.get()
            model.to(device)
            model.eval()
            retraining = False
            print("Updated model loaded!")
        
        # Detect and analyze faces
        faces = detect_faces(frame, face_analyzer)
        
        # Process each detected face
        for face in faces:
            if capturing_new_person:
                # Special mode for adding a new person
                draw_face_info(display_frame, face, f"Collecting: {new_person_name}", config.get_color('visualization.landmarks.color'))
                if show_landmarks:
                    draw_landmarks(display_frame, face, color=config.get_color('visualization.landmarks.color'))
            else:
                # Normal recognition mode
                label, color, looking = process_frame_for_recognition(
                    frame, face, face_analyzer, model, classes,
                    device, confidence_threshold, check_looking
                )
                
                # Draw bounding box and label
                draw_face_info(display_frame, face, label, color)
                
                # Draw landmarks if enabled
                if show_landmarks:
                    draw_landmarks(display_frame, face, color=color)
        
        # Calculate processing time and FPS
        process_time = (time.time() - start_time) * 1000  # ms
        current_fps = 1.0 / (time.time() - start_time) if (time.time() - start_time) > 0 else 30.0
        detection_fps = 0.9 * detection_fps + 0.1 * current_fps if detection_fps > 0 else current_fps
        
        # Additional information to display
        if capturing_new_person:
            extra_info = f"Adding {new_person_name} - Press 's' to save face, 'x' to finish"
        elif retraining:
            extra_info = "Retraining model in background..."
        else:
            extra_info = f"L: landmarks | C: camera-check | A: add person | R: retrain"
        
        draw_stats(display_frame, detection_fps, process_time, len(faces), extra_info)
        
        # Display the frame
        cv2.imshow("Face Recognition", display_frame)
        
        # Check for key presses
        key = cv2.waitKey(1)
        if key == ord(config.get('controls.quit')):
            break
        elif key == ord(config.get('controls.toggle_landmarks')):
            # Toggle landmarks display
            show_landmarks = not show_landmarks
            print(f"Landmarks display: {'ON' if show_landmarks else 'OFF'}")
        elif key == ord(config.get('controls.toggle_looking_detection')):
            # Toggle looking at camera check
            check_looking = not check_looking
            print(f"Looking at camera check: {'ON' if check_looking else 'OFF'}")
        elif key == ord('a') and not capturing_new_person and not retraining:
            # Start capturing a new person
            capturing_new_person = True
            print("Enter person's name:")
            new_person_name = input().strip()
            if new_person_name:
                person_dir = create_person_dir(new_person_name, DATASET_PATH)
                print(f"Ready to capture images for {new_person_name}. Press 's' to save faces.")
            else:
                capturing_new_person = False
                print("No name entered. Cancelled.")
        elif key == ord('s') and capturing_new_person:
            # Save faces for the new person
            person_dir = os.path.join(DATASET_PATH, new_person_name.lower().replace(" ", "_"))
            for face in faces:
                filename = save_face(frame, face, person_dir)
                print(f"Face saved: {filename}")
        elif key == ord('x') and capturing_new_person:
            # Finish capturing new person
            capturing_new_person = False
            print(f"Finished capturing {new_person_name}.")
        elif key == ord('r') and not retraining and not capturing_new_person:
            # Retrain the model in a background thread
            retraining = True
            train_thread = threading.Thread(target=retrain_model_thread, args=(model_queue,))
            train_thread.daemon = True
            train_thread.start()
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Main application
def main():
    # Check if requirements are met
    if not check_requirements():
        print("Please install the required packages first.")
        return
    
    # Create dataset directory if it doesn't exist
    os.makedirs(DATASET_PATH, exist_ok=True)
    
    # Main menu loop
    while True:
        clear_console()
        print_header("Face Recognition System")
        
        # Display model status
        if os.path.exists(MODEL_PATH):
            print("Model status: Trained")
            try:
                device = torch.device(config.get_device())
                model, classes = load_model(MODEL_PATH, device)
                if model is not None:
                    print(f"Available classes: {classes}")
            except Exception as e:
                print(f"Model error: {e}")
        else:
            print("Model status: Not trained")
        
        # Display dataset status
        person_counts = count_images_by_person(DATASET_PATH)
        if person_counts:
            print("\nDataset status:")
            for person, count in person_counts.items():
                print(f"- {person}: {count} images")
        else:
            print("\nDataset status: Empty")
        
        # Display menu
        print("\nSelect an option:")
        print("1. Detect and collect face images")
        print("2. Train recognition model")
        print("3. Evaluate recognition model")
        print("4. Run real-time recognition")
        print("5. Exit")
        
        # Get user choice
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            detect_and_collect_faces()
        elif choice == "2":
            train_recognition_model()
        elif choice == "3":
            evaluate_recognition_model()
        elif choice == "4":
            real_time_recognition()
        elif choice == "5":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")
            wait_key()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
