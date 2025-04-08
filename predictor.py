import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import argparse
import time
import os
from tqdm import tqdm

# Definition of MLP architecture, same as used in training
class FaceMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FaceMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # Flatten image to vector
        x = x.view(x.size(0), -1)
        return self.model(x)

# Dataset for test images (last 3 images from each class folder)
class FaceTestDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, classes, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = classes
        
        # For each class, look for test images (last 3) only
        for class_idx, cls in enumerate(self.classes):
            cls_path = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_path):
                print(f"Warning: Folder {cls_path} not found!")
                continue
            
            # Get all image files in the class folder
            image_files = []
            for file_name in os.listdir(cls_path):
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.pgm')):
                    image_files.append(file_name)
            
            # Sort the files to ensure consistent order
            image_files.sort()
            
            # For testing: use only the last 3 images
            if len(image_files) <= 3:
                print(f"Warning: Class {cls} has only {len(image_files)} images for testing")
                test_files = image_files
            else:
                test_files = image_files[-3:]
            
            # Add test files to dataset
            for file_name in test_files:
                self.image_paths.append(os.path.join(cls_path, file_name))
                self.labels.append(class_idx)
            
            print(f"Class {cls}: {len(test_files)} images for testing")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label, img_path  # Also return image path for reference

# Function to perform prediction for a single image
def predict_image(model, image_path, transform, device, classes):
    start_time = time.time()
    
    # Log image information
    img_size = os.path.getsize(image_path) / 1024  # KB
    print(f"Processing image: {os.path.basename(image_path)} ({img_size:.2f} KB)")
    
    # Load and process image
    try:
        image = Image.open(image_path)
        img_width, img_height = image.size
        print(f"Original dimensions: {img_width}x{img_height} pixels")
        
        # Preprocessing time
        preprocess_start = time.time()
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)  # add batch dimension
        image_tensor = image_tensor.to(device)
        preprocess_time = time.time() - preprocess_start
        print(f"Preprocessing completed in {preprocess_time*1000:.2f} ms")
        
        # Inference time
        inference_start = time.time()
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        inference_time = time.time() - inference_start
        print(f"Inference completed in {inference_time*1000:.2f} ms")
        
        # Get results
        predicted_idx = outputs.argmax(dim=1).item()
        
        # Show probabilities for all classes
        print("\nClassification results:")
        print("-" * 40)
        print(f"{'Class':<15} | {'Probability':<15} | {'Result'}")
        print("-" * 40)
        
        for i, (cls, prob) in enumerate(zip(classes, probabilities)):
            confidence = prob.item() * 100
            result_mark = "✓" if i == predicted_idx else " "
            print(f"{cls:<15} | {confidence:>13.2f}% | {result_mark}")
        
        total_time = time.time() - start_time
        print("\nTotal processing time: {:.2f} ms".format(total_time * 1000))
        
        return predicted_idx, probabilities.tolist()
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None, None

# Function to evaluate model on entire test set
def evaluate_model(model, dataset_path, classes, device):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Create test dataset
    test_dataset = FaceTestDataset(dataset_path, classes, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Evaluation statistics
    correct = 0
    total = 0
    class_correct = [0] * len(classes)
    class_total = [0] * len(classes)
    class_confidence_sum = [0.0] * len(classes)  # To accumulate confidence per class
    class_predictions = [0] * len(classes)  # Counter for how many times each class was predicted
    model.eval()
    
    print("\n" + "="*50)
    print(f"EVALUATING MODEL WITH {len(test_dataset)} TEST IMAGES (LAST 3 IMAGES FROM EACH CLASS)")
    print("="*50)
    
    with torch.no_grad():
        for images, labels, paths in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            # Calculate probabilities using softmax
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            confidence = probabilities[predicted.item()].item() * 100
            
            # General statistics
            total += labels.size(0)
            is_correct = (predicted == labels).item()
            correct += is_correct
            
            # Per-class statistics
            label = labels[0].item()
            class_total[label] += 1
            if is_correct:
                class_correct[label] += 1
            
            # Accumulate confidence for predicted class
            predicted_class = predicted.item()
            class_confidence_sum[predicted_class] += confidence
            class_predictions[predicted_class] += 1
            
            # Display result for each image with confidence
            result_mark = "✓" if is_correct else "✗"
            filename = os.path.basename(paths[0])
            print(f"{result_mark} {filename} - Predicted: {classes[predicted.item()]} ({confidence:.2f}%) | Actual: {classes[label]}")
    
    # Display final results
    accuracy = 100 * correct / total
    print("\n" + "="*50)
    print(f"OVERALL ACCURACY: {accuracy:.2f}% ({correct}/{total})")
    print("="*50)
    
    # Display per-class results
    print("\nACCURACY BY CLASS:")
    print("-" * 60)
    print(f"{'Class':<15} | {'Accuracy':<15} | {'Average Confidence':<20} | {'Details'}")
    print("-" * 60)
    for i in range(len(classes)):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            
            # Calculate average confidence for predictions of this class
            avg_confidence = class_confidence_sum[i] / class_predictions[i] if class_predictions[i] > 0 else 0
            
            print(f"{classes[i]:<15} | {class_acc:>6.2f}% | {avg_confidence:>18.2f}% | ({class_correct[i]}/{class_total[i]})")
    
    return accuracy

def main():
    # Hardcoded configurations
    model_path = "face_mlp.pth"  # Path to trained model
    dataset_path = "dataset/"  # Path to dataset (used for evaluation)
    
    print("\n" + "="*50)
    print("FACIAL RECOGNITION SYSTEM")
    print("="*50)
    
    # Define device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model and classes
    print(f"Loading model from: {model_path}")
    try:
        model_load_start = time.time()
        
        # Load checkpoint containing model state and classes
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract class information
        if isinstance(checkpoint, dict) and 'classes' in checkpoint:
            classes = checkpoint['classes']
            model_state = checkpoint['model_state_dict']
            print(f"Loaded {len(classes)} classes from model file")
        else:
            # Fallback in case model file doesn't contain classes
            print("Model file doesn't contain class information. Scanning dataset directory...")
            classes = []
            if os.path.exists(dataset_path):
                for item in os.listdir(dataset_path):
                    item_path = os.path.join(dataset_path, item)
                    if os.path.isdir(item_path):
                        classes.append(item)
            classes.sort()
            print(f"Found {len(classes)} classes in dataset directory")
            model_state = checkpoint
        
        # Initialize model
        input_size = 92 * 112 * 1  # grayscale 92x112
        num_classes = len(classes)
        model = FaceMLP(input_size, num_classes)
        
        # Load weights
        model.load_state_dict(model_state)
        model_load_time = time.time() - model_load_start
        print(f"Model loaded in {model_load_time*1000:.2f} ms")
        
        # Count model parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
        print(f"Available classes: {classes}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    model.to(device)
    model.eval()
    
    # Define transformations: convert to grayscale, tensor and normalize
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalization for grayscale images
    ])
    
    # Menu of options
    print("\nChoose an option:")
    print("1. Evaluate model on all test images (last 3 from each class)")
    print("2. Test a specific image")
    choice = input("Option (1 or 2): ").strip()
    
    if choice == "1":
        # Evaluate model on all test images
        evaluate_model(model, dataset_path, classes, device)
    elif choice == "2":
        # Test a specific image
        image_path = input("Enter the image path to test: ")
        
        print("\n" + "-"*50)
        # Perform prediction for the provided image
        predicted_idx, probabilities = predict_image(model, image_path, transform, device, classes)
        
        if predicted_idx is not None:
            predicted_label = classes[predicted_idx]
            confidence = probabilities[predicted_idx] * 100
            
            print("\n" + "="*50)
            print(f"FINAL RESULT: {predicted_label}")
            print(f"Confidence: {confidence:.2f}%")
            print("="*50 + "\n")
    else:
        print("Invalid option!")

if __name__ == "__main__":
    main()
