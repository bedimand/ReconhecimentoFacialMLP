import os
import cv2
import argparse
import time
from datetime import datetime
import sys

# Import necessary modules from the project
from src.utils.config import config
from src.utils.utils import create_person_dir, print_header
from src.face.face_detection import initialize_face_analyzer, detect_faces, save_face
from src.utils.visualization import draw_landmarks, draw_stats

# Get paths from config
DATASET_PATH = config.get_dataset_dir()

def process_images(input_folder, person_name=None, show_visualization=True, save_visualization=False, auto_continue=False):
    """
    Process all images in the specified folder, detect faces and save them to the dataset.
    
    Args:
        input_folder (str): Path to folder containing images to process
        person_name (str, optional): Name of the person in the images
        show_visualization (bool): Whether to show visualization of detected faces
        save_visualization (bool): Whether to save visualization images
        auto_continue (bool): Whether to automatically process all images without waiting for keypresses
    """
    # Initialize face analyzer
    print("Initializing face detection...")
    face_analyzer = initialize_face_analyzer()
    if face_analyzer is None:
        print("Failed to initialize face detection. Please make sure models are downloaded.")
        return
    
    # Create person directory if name is provided
    if person_name:
        person_dir = create_person_dir(person_name, DATASET_PATH)
    else:
        person_dir = os.path.join(DATASET_PATH, "unknown")
        os.makedirs(person_dir, exist_ok=True)
    
    # Determine cleaned label name and start timestamp for this collection session
    label_name = os.path.basename(person_dir)
    collection_ts = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return
    
    # Get list of image files
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [
        f for f in os.listdir(input_folder) 
        if os.path.isfile(os.path.join(input_folder, f)) and 
        any(f.lower().endswith(ext) for ext in valid_extensions)
    ]
    
    if not image_files:
        print(f"No image files found in '{input_folder}'.")
        return
    
    print(f"Found {len(image_files)} images to process.")
    
    # Create visualization output folder if needed
    if save_visualization:
        viz_folder = os.path.join(input_folder, "visualization")
        os.makedirs(viz_folder, exist_ok=True)
        print(f"Visualization images will be saved to: {viz_folder}")
    
    # Display instructions if showing visualization and not in auto mode
    if show_visualization and not auto_continue:
        print("\nVisualization window will open. Press:")
        print("  SPACE or ENTER to continue to next image")
        print("  'q' to stop processing and exit")
        print("  'a' to process all remaining images without pausing")
        print("\nPress any key to begin processing...")
        cv2.namedWindow("Face Detection", cv2.WINDOW_NORMAL)
        cv2.waitKey(0)
    elif auto_continue:
        print("Auto-processing enabled. Processing all images...")
    
    # Process each image
    face_count = 0
    total_time = 0
    auto_mode = auto_continue  # Start in auto mode if specified
    
    for i, img_file in enumerate(image_files):
        # Print progress
        progress = (i + 1) / len(image_files) * 100
        progress_bar = f"[{'=' * int(progress // 2)}>{' ' * (50 - int(progress // 2))}]"
        print(f"\r{progress_bar} {progress:.1f}% ({i+1}/{len(image_files)})", end="")
        sys.stdout.flush()
        
        img_path = os.path.join(input_folder, img_file)
        
        # Read image
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"\n  Error: Could not read image {img_file}. Skipping.")
            continue
        
        # Detect faces
        start_time = time.time()
        faces = detect_faces(frame, face_analyzer)
        process_time = (time.time() - start_time) * 1000  # ms
        total_time += process_time
        
        # Only print full details if not in progress bar mode
        if not auto_mode:
            print(f"\nProcessing: {img_file} ({i+1}/{len(image_files)})")
            print(f"  Detected {len(faces)} faces in {process_time:.1f}ms")
        
        if len(faces) == 0:
            if not auto_mode:
                print("  No faces detected in this image. Skipping.")
            continue
        
        # Create visualization
        if (show_visualization or save_visualization) and len(faces) > 0:
            display_frame = frame.copy()
            
            # Draw detected faces
            for face in faces:
                # Get bounding box
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                
                # Draw bounding box
                cv2.rectangle(
                    display_frame, 
                    (x1, y1), 
                    (x2, y2), 
                    config.get_color('visualization.bounding_box.high_confidence_color'), 
                    2
                )
                
                # Draw facial landmarks
                draw_landmarks(
                    display_frame, 
                    face, 
                    color=config.get_color('visualization.landmarks.color')
                )
            
            # Draw stats
            image_info = f"File: {img_file} ({i+1}/{len(image_files)})"
            draw_stats(display_frame, 0, process_time, len(faces), image_info)
            
            # Save visualization
            if save_visualization:
                viz_path = os.path.join(viz_folder, f"viz_{os.path.basename(img_file)}")
                cv2.imwrite(viz_path, display_frame)
            
            # Show visualization if not in auto mode
            if show_visualization and not auto_mode:
                cv2.imshow("Face Detection", display_frame)
                # Wait for key with instructions
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q'):
                    print("\nProcessing stopped by user.")
                    break
                elif key == ord('a'):
                    print("\nAuto-continuing enabled. Processing all remaining images...")
                    auto_mode = True
        
        # Save each detected face
        for j, face in enumerate(faces):
            filename = save_face(frame, face, person_dir, label_name, collection_ts)
            if not auto_mode:
                print(f"  Saved face {j+1}/{len(faces)}: {filename}")
            face_count += 1
    
    # Cleanup and clear progress bar
    print("\n")  # Move past the progress bar
    if show_visualization:
        cv2.destroyAllWindows()
    
    # Summary
    if len(image_files) > 0:
        avg_time = total_time / len(image_files)
        print(f"\nProcessing complete:")
        print(f"  Processed {len(image_files)} images")
        print(f"  Extracted {face_count} faces")
        print(f"  Average processing time: {avg_time:.1f}ms per image")
        print(f"  Faces saved to: {person_dir}")
    
def main():
    """Main function to parse arguments and run face capture from images"""
    parser = argparse.ArgumentParser(description="Process images and extract faces for the face recognition dataset")
    
    parser.add_argument("input_folder", help="Path to folder containing images to process")
    parser.add_argument("-n", "--name", help="Name of the person in the images")
    parser.add_argument("--no-viz", action="store_true", help="Don't show visualization window")
    parser.add_argument("--save-viz", action="store_true", help="Save visualization images")
    parser.add_argument("--auto", action="store_true", help="Automatically process all images without pausing")
    
    args = parser.parse_args()
    
    print_header("Face Capture from Images")
    process_images(
        args.input_folder, 
        person_name=args.name, 
        show_visualization=not args.no_viz,
        save_visualization=args.save_viz,
        auto_continue=args.auto
    )
    
    print("\nPress Enter to exit...")
    input()

if __name__ == "__main__":
    main()
