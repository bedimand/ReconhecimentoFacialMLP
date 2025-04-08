import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import time
from tqdm import tqdm
import cv2
import numpy as np

from src.model import FaceMLP, save_model
from src.preprocessing import preprocess_face

def get_classes(dataset_path):
    """
    Get class names from dataset directory structure
    
    Args:
        dataset_path: Path to dataset
        
    Returns:
        List of class names
    """
    classes = []
    if os.path.exists(dataset_path):
        # Get all directories in the dataset path
        for item in os.listdir(dataset_path):
            item_path = os.path.join(dataset_path, item)
            if os.path.isdir(item_path):
                classes.append(item)
    
    # Sort classes for consistent ordering
    classes.sort()
    
    if not classes:
        print(f"WARNING: No class folders found in {dataset_path}")
    else:
        print(f"Found {len(classes)} classes: {classes}")
    
    return classes

class FaceDataset(Dataset):
    """Dataset for facial recognition training and testing"""
    def __init__(self, root_dir, classes, transform=None, is_train=True):
        self.root_dir = root_dir
        self.classes = classes
        self.transform = transform
        self.is_train = is_train
        self.image_paths = []
        self.labels = []
        
        # For each class, look for images in corresponding folder
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
            
            # For training: use all images except the last 3
            # For testing: use only the last 3 images
            if len(image_files) <= 3:
                print(f"Warning: Class {cls} has only {len(image_files)} images, need at least 4 for train/test split")
                if self.is_train and len(image_files) > 0:
                    # If training and there's at least 1 image, use 1 fewer than available
                    train_files = image_files[:max(1, len(image_files)-1)]
                    for file_name in train_files:
                        self.image_paths.append(os.path.join(cls_path, file_name))
                        self.labels.append(class_idx)
                elif not self.is_train and len(image_files) > 0:
                    # If testing and there's at least 1 image, use the last one
                    test_files = image_files[-1:]
                    for file_name in test_files:
                        self.image_paths.append(os.path.join(cls_path, file_name))
                        self.labels.append(class_idx)
            else:
                # Normal case: at least 4 images available
                if self.is_train:
                    # Training: all but last 3
                    train_files = image_files[:-3]
                    for file_name in train_files:
                        self.image_paths.append(os.path.join(cls_path, file_name))
                        self.labels.append(class_idx)
                else:
                    # Testing: last 3 only
                    test_files = image_files[-3:]
                    for file_name in test_files:
                        self.image_paths.append(os.path.join(cls_path, file_name))
                        self.labels.append(class_idx)
            
            # Log how many images are being used
            if self.is_train:
                print(f"Class {cls}: {len(image_files) - min(3, len(image_files))} images for training")
            else:
                print(f"Class {cls}: {min(3, len(image_files))} images for testing")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Read image using cv2 to apply preprocessing
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not read image: {img_path}")
        
        # Apply preprocessing
        processed_img = preprocess_face(image)
        
        # Convert to PIL Image for the transform
        image = Image.fromarray(processed_img)
        
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

def train_model(dataset_path, model_save_path, num_epochs=20, batch_size=16, learning_rate=0.001):
    """
    Train the face recognition model
    
    Args:
        dataset_path: Path to dataset directory
        model_save_path: Path to save the trained model
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
    
    Returns:
        Trained model and class names
    """
    # Get classes from dataset folder structure
    classes = get_classes(dataset_path)
    if not classes:
        print("Error: No classes found in dataset. Aborting training.")
        return None, []
    
    # Transformations: convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalization for grayscale images
    ])
    
    # Create dataset and dataloader (training only)
    train_dataset = FaceDataset(dataset_path, classes, transform=transform, is_train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    if len(train_dataset) == 0:
        print("Error: No training images found. Aborting training.")
        return None, []
    
    num_classes = len(classes)
    input_size = 92 * 112 * 1  # grayscale 92x112 images
    model = FaceMLP(input_size, num_classes)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Define device: GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Log initial information
    print("\n" + "="*50)
    print(f"Starting training with {len(train_dataset)} images")
    print(f"Classes: {classes}")
    print(f"Device: {device}")
    print("="*50 + "\n")
    
    # Start time
    start_time = time.time()
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_start = time.time()
        
        # Progress bar for each epoch
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100 * correct / total:.2f}%"
            })
        
        # Epoch statistics
        epoch_time = time.time() - epoch_start
        avg_loss = running_loss / len(train_dataloader)
        accuracy = 100 * correct / total
        
        print(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s - "
              f"Loss: {avg_loss:.4f} - Accuracy: {accuracy:.2f}%")
    
    # Total training time
    total_time = time.time() - start_time
    print("\n" + "="*50)
    print(f"Training completed in {total_time:.2f} seconds")
    
    # Save the trained model and class information
    save_model(model, classes, model_save_path)
    
    return model, classes 