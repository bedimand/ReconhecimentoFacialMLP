import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import time
import os
import matplotlib.pyplot as plt

from src.model import load_model
from src.face_detection import detect_faces, extract_face, is_looking_at_camera
from src.preprocessing import preprocess_face

def get_transform():
    """Get transform for face recognition"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

def recognize_face(model, face_img, classes, device, transform=None):
    """
    Recognize a face in an image
    
    Args:
        model: Face recognition model
        face_img: Preprocessed face image
        classes: List of class names
        device: Torch device
        transform: Optional transform to apply
        
    Returns:
        predicted_idx: Index of predicted class
        confidence: Confidence score
        probabilities: List of probabilities for all classes
    """
    if transform is None:
        transform = get_transform()
    
    # Convert to PIL Image if it's a numpy array
    if isinstance(face_img, np.ndarray):
        face_img = Image.fromarray(face_img)
    
    # Apply transform and add batch dimension
    face_tensor = transform(face_img).unsqueeze(0).to(device)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(face_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
    # Get predictions
    predicted_idx = outputs.argmax(dim=1).item()
    confidence = probabilities[predicted_idx].item() * 100
    
    return predicted_idx, confidence, probabilities.tolist()

def evaluate_model(model, dataset_path, classes, device):
    """
    Evaluate model on test images (reserved from each class)
    
    Args:
        model: Face recognition model
        dataset_path: Path to dataset
        classes: List of class names
        device: Torch device
        
    Returns:
        accuracy: Overall accuracy percentage
    """
    from src.training import FaceDataset
    from sklearn.metrics import confusion_matrix, classification_report
    
    # Define transform
    transform = get_transform()
    
    # Create test dataset
    test_dataset = FaceDataset(dataset_path, classes, transform=transform, is_train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print(f"\nEvaluating model with {len(test_dataset)} test images...")
    
    # Evaluation statistics
    correct = 0
    total = 0
    class_correct = [0] * len(classes)
    class_total = [0] * len(classes)
    
    # Store all predictions and true labels for confusion matrix
    all_predictions = []
    all_labels = []
    
    # Model in evaluation mode
    model.eval()
    
    # Process each test image
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            # Store for confusion matrix
            all_predictions.append(predicted.item())
            all_labels.append(labels.item())
            
            # Update statistics
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class statistics
            label = labels[0].item()
            class_total[label] += 1
            if predicted[0].item() == label:
                class_correct[label] += 1
    
    # Calculate accuracy
    accuracy = 100 * correct / total if total > 0 else 0
    
    # Display results
    print(f"\nOverall accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    # Print classification report
    try:
        report = classification_report(all_labels, all_predictions, target_names=classes, zero_division=0)
        print("\nDetailed Classification Report:")
        print(report)
    except ImportError:
        print("\nPlease install scikit-learn for detailed classification report")
    
    # Print per-class accuracy (for platforms without sklearn)
    print("\nAccuracy by class:")
    for i in range(len(classes)):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            print(f"{classes[i]}: {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})")
    
    # Plot confusion matrix if sklearn is available
    try:
        # Calculate confusion matrix
        cm = confusion_matrix(all_labels, all_predictions, labels=range(len(classes)))
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45, ha='right')
        plt.yticks(tick_marks, classes)
        
        # Add text annotations to the matrix
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save the figure
        plt.savefig('confusion_matrix.png')
        print("Confusion matrix saved to confusion_matrix.png")
    except ImportError:
        print("Please install scikit-learn and matplotlib for confusion matrix visualization")
    
    return accuracy

def process_frame_for_recognition(frame, face, face_analyzer, model, classes, 
                                 device, confidence_threshold=80.0, check_looking=True):
    """
    Process a single detected face for recognition
    
    Args:
        frame: Input camera frame
        face: Detected face from InsightFace
        face_analyzer: InsightFace analyzer
        model: Face recognition model
        classes: List of class names
        device: Torch device
        confidence_threshold: Threshold for recognition confidence
        check_looking: Whether to check if person is looking at camera
        
    Returns:
        label: Recognition label with confidence
        color: Color for drawing (based on confidence/looking)
        looking_at_camera: Whether person is looking at camera
    """
    # Check if person is looking at camera
    looking_at_camera = not check_looking or is_looking_at_camera(face)
    
    # Only process face recognition if looking at camera or check is disabled
    if looking_at_camera:
        # Extract and preprocess face
        face_img = extract_face(frame, face)
        
        # Convert to PIL Image
        pil_img = Image.fromarray(face_img)
        
        # Get transform
        transform = get_transform()
        
        # Recognize face
        predicted_idx, confidence, _ = recognize_face(
            model, pil_img, classes, device, transform
        )
        
        # Get predicted class and confidence
        predicted_label = classes[predicted_idx]
        
        # Check if the predicted label is from the 'unknown' folder or has low confidence
        if predicted_label.startswith('unknown_') or confidence < confidence_threshold:
            color = (0, 0, 255)  # Red for unknown
            label = f"Desconhecido: {confidence:.1f}%"
        else:
            # Draw bounding box - green for high confidence
            color = (0, 255, 0)
            label = f"{predicted_label}: {confidence:.1f}%"
    else:
        # If not looking at camera, display message and use different color
        color = (0, 0, 255)  # Red for not looking
        label = "Look at the camera"
    
    return label, color, looking_at_camera
