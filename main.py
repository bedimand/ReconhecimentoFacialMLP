import os
import cv2
import torch
import numpy as np
import time
import json
from datetime import datetime

from src.utils.config import config
from src.utils.utils import (
    clear_console, 
    print_header,
    wait_key,
    create_person_dir,
    count_images_by_person
)
from src.face.face_detection import (
    initialize_face_analyzer, 
    detect_faces, 
    save_face,
    extract_face
)
from src.frame_processor import process_frame_for_recognition
from src.model.evaluation import evaluate_model, print_evaluation_results, plot_confusion_matrix
from src.utils.visualization import (
    draw_landmarks, 
    draw_stats, 
    draw_five_landmarks, 
    draw_nose_centered_crop, 
    draw_crosshair_center,
    draw_recognition_results
)
from src.model.train import train_model
from src.data.preprocessing_dataset import preprocess_dataset
from src.model.model import load_model
from src.face.preprocessing import PreprocessModule
from src.utils.capture_faces import process_images

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
    
    # Determine cleaned label name and start timestamp for this collection session
    label_name = os.path.basename(person_dir)
    collection_ts = datetime.now().strftime("%Y%m%d%H%M%S")
    
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
                filename = save_face(frame, face, person_dir, label_name, collection_ts)
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
                filename = save_face(frame, face, person_dir, label_name, collection_ts)
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

# Function to capture faces from images in a folder
def capture_faces_from_folder():
    print_header("Capture Faces from Image Folder")
    
    # Ask for the folder path
    print("Enter the folder path containing the images:")
    folder_path = input().strip()
    
    if not folder_path:
        print("No folder path provided.")
        wait_key()
        return
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        wait_key()
        return
    
    # Ask for person's name
    print("Enter the person's name (or press Enter for unnamed faces):")
    person_name = input().strip()
    
    # Ask for visualization preferences
    print("\nVisualization options:")
    print("1. Show each image and wait for keypress (interactive)")
    print("2. Process all images without stopping (automatic)")
    print("3. No visualization (fastest)")
    viz_choice = input("Enter your choice (1-3): ").strip()
    
    auto_process = viz_choice == "2"
    show_viz = viz_choice in ("1", "2", "")
    
    # Ask about saving visualization images
    print("\nSave visualization images with detected faces? (y/n):")
    save_viz = input().strip().lower() in ('y', 'yes', 'true', '1')
    
    print("\nStarting face capture from images...")
    print("This will use the same nose-centered approach as the real-time capture.")
    print("Images will be saved to your dataset folder.")
    
    # Process the images
    process_images(
        folder_path,
        person_name=person_name if person_name else None,
        show_visualization=show_viz and not auto_process,
        save_visualization=save_viz,
        auto_continue=auto_process
    )
    
    wait_key()

# Function to train the recognition model
def train_recognition_model():
    print_header("Train Face Recognition Model")
    print("[LOG] Starting train_recognition_model")
    
    print("[LOG] Checking if dataset exists at:", DATASET_PATH)
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset path {DATASET_PATH} not found!")
        wait_key()
        return
    print("[LOG] Dataset directory found")
    
    print("[LOG] Counting images by person")
    person_counts = count_images_by_person(DATASET_PATH)
    
    print(f"[LOG] Found {len(person_counts)} person(s) in dataset")
    if not person_counts:
        print("No person folders found in dataset!")
        wait_key()
        return
    print("[LOG] person_counts:", person_counts)
    
    print("Available people in dataset:")
    for person, count in person_counts.items():
        print(f"- {person}: {count} images")
    
    print("[LOG] Retrieving training parameters from config")
    num_epochs = config.get('training.num_epochs')
    batch_size = config.get('training.batch_size')
    learning_rate = config.get('training.learning_rate')
    
    print("\nTraining parameters:")
    print(f"Number of epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    
    print("\nStarting training...")
    
    print(f"[LOG] Calling train_model with epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}")
    model, classes = train_model(
        DATASET_PATH, 
        MODEL_PATH, 
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    print("[LOG] train_model returned")
    
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
    
    # Use preprocessed dataset if available, else raw
    preprocessed_dir = DATASET_PATH.rstrip(os.sep) + '_preprocessed'
    if os.path.exists(preprocessed_dir):
        eval_dataset_path = preprocessed_dir
        print(f"[LOG] Using preprocessed dataset at {preprocessed_dir} for evaluation")
    else:
        eval_dataset_path = DATASET_PATH
        print(f"[LOG] Preprocessed dataset not found, evaluating on raw data at {DATASET_PATH}")
    
    # Load manifest to exclude training images
    manifest = {}
    manifest_path = os.path.join(eval_dataset_path, 'manifest.json')
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as mf:
            manifest = json.load(mf)
        print("[LOG] Loaded manifest for evaluation, excluding training images")
    else:
        print(f"[LOG] No manifest found at {manifest_path}, evaluating on all images")
    
    # Evaluate model
    accuracy, all_predictions, all_labels, class_accuracies = evaluate_model(model, eval_dataset_path, classes, device, manifest=manifest)
    
    # Print detailed results
    print_evaluation_results(accuracy, classes, all_labels, all_predictions, class_accuracies)
    
    # Generate confusion matrix visualization
    plot_confusion_matrix(classes, all_labels, all_predictions)
    
    wait_key()

# Function for real-time face recognition
def real_time_recognition(model=None, classes=None):
    print_header("Real-time Face Recognition")
    
    # Set fixed random seed for reproducibility
    seed = config.get('training.seed', 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Set random seed to {seed} for reproducible results")
    
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
    print("Press 'n' to toggle nose-centered crop visualization")
    print("Press 'f' to toggle 5-point landmarks visualization")
    print(f"Press '{config.get('controls.quit')}' to quit")
    
    # Main loop variables
    detection_fps = 0
    show_landmarks = config.get('visualization.landmarks.enabled', True)
    check_looking = config.get('recognition.looking_detection.enabled', True)
    confidence_threshold = config.get('recognition.confidence_threshold', 80.0)
    show_nose_centered = False  # Toggle for showing nose-centered crop
    show_five_landmarks = False  # Toggle for showing 5-point landmarks
    
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
        
        # Detect and analyze faces
        faces = detect_faces(frame, face_analyzer)
        
        # Process each detected face
        for face in faces:
            # If showing 5-point landmarks is enabled
            if show_five_landmarks:
                display_frame = draw_five_landmarks(display_frame, face, show_labels=True)
            
            # If showing nose-centered crop is enabled
            if show_nose_centered:
                display_frame = draw_nose_centered_crop(display_frame, face)
                # Extract the nose-centered crop for demonstration
                nose_crop = extract_face(frame, face)
                # Add crosshair to show center
                nose_crop = draw_crosshair_center(nose_crop)
                # Show the crop in a separate window
                cv2.imshow("Nose-Centered Crop (128x128)", nose_crop)
            
            # Normal recognition mode
            result = process_frame_for_recognition(
                frame, face, model, classes,
                device, confidence_threshold, check_looking
            )
            
            # Draw recognition results on display frame
            draw_recognition_results(display_frame, face, result)
        
        # Calculate processing time and FPS
        process_time = (time.time() - start_time) * 1000  # ms
        current_fps = 1.0 / (time.time() - start_time) if (time.time() - start_time) > 0 else 30.0
        detection_fps = 0.9 * detection_fps + 0.1 * current_fps if detection_fps > 0 else current_fps
        
        # Additional information to display
        extra_info = "L: landmarks | C: camera-check | N: nose-crop | F: 5-points"
        
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
        elif key == ord('n'):
            # Toggle nose-centered crop visualization
            show_nose_centered = not show_nose_centered
            print(f"Nose-centered crop visualization: {'ON' if show_nose_centered else 'OFF'}")
            if not show_nose_centered:
                cv2.destroyWindow("Nose-Centered Crop (128x128)")
        elif key == ord('f'):
            # Toggle 5-point landmarks visualization
            show_five_landmarks = not show_five_landmarks
            print(f"5-point landmarks visualization: {'ON' if show_five_landmarks else 'OFF'}")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def preprocess_and_export_image():
    print_header("Preprocess and Export Image")
    # Prompt the user to enter the image path manually
    file_path = input("Enter the path to the image you want to preprocess: ").strip()
    if not file_path:
        print("No file path provided.")
        wait_key()
        return

    # Read the image
    import cv2
    image = cv2.imread(file_path)
    if image is None:
        print("Failed to load image.")
        wait_key()
        return

    # Preprocess the image
    preprocessor = PreprocessModule(target_size=(128, 128))
    processed = preprocessor.preprocess_face(image)

    # Convert tensor to numpy and save as image (invert ImageNet normalization)
    import torch
    if isinstance(processed, torch.Tensor):
        # Detach and move to CPU
        arr = processed.detach().cpu().numpy()  # shape (C, H, W)
        # ImageNet mean and std for normalization
        mean = np.array([0.485, 0.456, 0.406])[:, None, None]
        std = np.array([0.229, 0.224, 0.225])[:, None, None]
        # Invert normalization: x = (x * std) + mean
        arr = (arr * std) + mean
        # Clip to [0,1]
        arr = np.clip(arr, 0, 1)
        # Convert from (C, H, W) to (H, W, C)
        processed = arr.transpose(1, 2, 0)
        # Convert to 0-255 uint8
        processed = (processed * 255).astype('uint8')

    # Save the processed image
    out_path = file_path.rsplit('.', 1)[0] + '_preprocessed.png'
    cv2.imwrite(out_path, processed)
    print(f"Preprocessed image saved to: {out_path}")
    wait_key()

def preprocess_all_images():
    # Bulk preprocess entire dataset
    print_header("Preprocess All Images")
    dataset_dir = DATASET_PATH
    preprocessed_dir = dataset_dir.rstrip(os.sep) + '_preprocessed'
    print(f"[LOG] Preprocessing all images from {dataset_dir} to {preprocessed_dir}")
    # Preprocess all images without sampling
    preprocess_dataset(dataset_dir, preprocessed_dir, target_size=(128,128))
    print("[LOG] Preprocessing complete.")
    wait_key()
    return

# Main application
def main():
    
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
        print("1. Detect and collect face images (webcam)")
        print("2. Capture faces from image folder")
        print("3. Train recognition model")
        print("4. Evaluate recognition model")
        print("5. Run real-time recognition")
        print("6. Preprocess and export an image")
        print("7. Preprocess all images")
        print("8. Exit")
        
        # Get user choice
        choice = input("\nEnter your choice (1-8): ").strip()
        
        if choice == "1":
            detect_and_collect_faces()
        elif choice == "2":
            capture_faces_from_folder()
        elif choice == "3":
            train_recognition_model()
        elif choice == "4":
            evaluate_recognition_model()
        elif choice == "5":
            real_time_recognition()
        elif choice == "6":
            preprocess_and_export_image()
        elif choice == "7":
            preprocess_all_images()
        elif choice == "8":
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
