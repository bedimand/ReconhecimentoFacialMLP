import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset, DataLoader
from src.face.face_recognition import TransformFactory

def evaluate_model(model, dataset_path, classes, device, manifest=None):
    """
    Evaluate model on test images and generate performance metrics
    
    Args:
        model: Face recognition model
        dataset_path: Path to dataset
        classes: List of class names
        device: Torch device
        manifest: Optional manifest dict of training images to exclude
        
    Returns:
        accuracy: Overall accuracy percentage
        confusion_mat: Confusion matrix
        class_accuracies: Per-class accuracy dict
    """
    from src.data.datasets import FaceDataset
    
    # Get centralized transform
    transform = TransformFactory.get_face_transform()
    
    # Create test dataset
    test_dataset = FaceDataset(dataset_path, classes, transform=transform, is_train=False)
    
    # Exclude training images if manifest provided
    if manifest:
        filtered_indices = []
        for idx, path in enumerate(test_dataset.image_paths):
            cls_name = classes[test_dataset.labels[idx]]
            filename = os.path.basename(path)
            if cls_name in manifest and filename in manifest[cls_name]:
                continue
            filtered_indices.append(idx)
        print(f"[LOG] Excluding {len(test_dataset.image_paths) - len(filtered_indices)} training images from evaluation")
        test_dataset = Subset(test_dataset, filtered_indices)
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
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
    
    # Calculate per-class accuracies
    class_accuracies = {}
    for i in range(len(classes)):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            class_accuracies[classes[i]] = class_acc
    
    return accuracy, all_predictions, all_labels, class_accuracies

def print_evaluation_results(accuracy, classes, all_labels, all_predictions, class_accuracies):
    """
    Print evaluation results to console
    """
    # Display results
    print(f"\nOverall accuracy: {accuracy:.2f}%")
    
    # Try to print classification report
    try:
        from sklearn.metrics import classification_report
        report = classification_report(all_labels, all_predictions, target_names=classes, zero_division=0)
        print("\nDetailed Classification Report:")
        print(report)
    except ImportError:
        print("\nPlease install scikit-learn for detailed classification report")
    
    # Print per-class accuracy
    print("\nAccuracy by class:")
    for class_name, acc in class_accuracies.items():
        print(f"{class_name}: {acc:.2f}%")

def plot_confusion_matrix(classes, all_labels, all_predictions, output_path="confusion_matrix.png"):
    """
    Plot and save a confusion matrix visualization
    """
    try:
        from sklearn.metrics import confusion_matrix
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
        plt.savefig(output_path)
        print(f"Confusion matrix saved to {output_path}")
        return cm
    except ImportError:
        print("Please install scikit-learn and matplotlib for confusion matrix visualization")
        return None 