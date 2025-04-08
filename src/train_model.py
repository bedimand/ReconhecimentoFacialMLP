import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import argparse
import time
from tqdm import tqdm
import cv2
import numpy as np

# Dynamic class detection based on folder names in dataset
def get_classes(dataset_path):
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

# Dataset: expects images to be organized in folders with class names
class FaceDataset(Dataset):
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
                print(f"Class {cls}: {len(self.image_paths) - sum(self.labels[:class_idx])} images for training")
            else:
                print(f"Class {cls}: {len(self.image_paths) - sum(1 for l in self.labels if l < class_idx)} images for testing")
    
    def __len__(self):
        return len(self.image_paths)
    
    def preprocess_image(self, image):
        """
        Apply the same preprocessing steps as in real_time_recognition.py
        """
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Histogram Equalization
        image = cv2.equalizeHist(image)
        
        # Edge Enhancement
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Combine edges
        edges = np.sqrt(sobelx**2 + sobely**2)
        edges = np.uint8(edges)
        
        # Enhance original image with edges
        alpha = 0.3  # Edge enhancement factor
        image = cv2.addWeighted(image, 1.0, edges, alpha, 0)
        
        return image
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Read image using cv2 to apply preprocessing
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not read image: {img_path}")
        
        # Apply preprocessing
        image = self.preprocess_image(image)
        
        # Convert to PIL Image for the transform
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# Definition of MLP for facial recognition
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

# Training function and model export
def train_model(dataset_path, model_save_path, num_epochs=10, batch_size=32, learning_rate=0.001):
    # Get classes from dataset folder structure
    classes = get_classes(dataset_path)
    if not classes:
        print("Error: No classes found in dataset. Aborting training.")
        return
    
    # Transformations: resize to maintain original 92x112 size, convert to tensor and normalize
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalization for grayscale images
    ])
    
    # Create dataset and dataloader (training only)
    train_dataset = FaceDataset(dataset_path, classes, transform=transform, is_train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
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
    print(f"Starting training with {len(train_dataset)} images (using all except last 3 from each class)")
    print(f"Classes: {classes}")
    print(f"Device: {device}")
    
    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")
    print("="*50 + "\n")
    
    # Save class names along with the model for future reference
    class_info = {'classes': classes}
    
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
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': classes
    }, model_save_path)
    print(f"Trained model saved to: {model_save_path}")
    print("="*50)

# Main block with hardcoded variables instead of command line arguments
if __name__ == "__main__":
    # Hardcoded configurations
    dataset_path = "dataset/"  # Path to dataset
    model_save_path = "face_mlp.pth"  # Path to save trained model
    num_epochs = 20  # Number of epochs (increased for better learning)
    batch_size = 16  # Batch size (reduced due to smaller dataset)
    learning_rate = 0.001  # Learning rate
    
    # Display configurations to be used
    print("\n" + "="*50)
    print("STARTING TRAINING WITH CONFIGURATIONS:")
    print(f"Dataset: {dataset_path}")
    print(f"Model will be saved to: {model_save_path}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print("="*50 + "\n")
    
    # Call training function with defined values
    train_model(dataset_path, model_save_path, num_epochs, batch_size, learning_rate)
