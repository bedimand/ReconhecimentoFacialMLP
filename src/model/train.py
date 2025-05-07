import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import random
import json
from src.model.model import FaceRecognitionMLP, save_model
from src.data.datasets import FaceDataset, get_classes
from src.face.face_recognition import TransformFactory
from src.utils.utils import wait_key, print_header
from src.utils.config import config

def train_model(dataset_path, model_save_path, num_epochs=None, batch_size=None, learning_rate=None):
    """
    Train the face recognition model with the given dataset
    
    Args:
        dataset_path: Path to the dataset folder
        model_save_path: Path to save the trained model
        num_epochs: Number of epochs to train for (if None, pulled from config)
        batch_size: Batch size for training (if None, pulled from config)
        learning_rate: Learning rate for optimizer (if None, pulled from config)
        
    Returns:
        model: Trained model
        classes: List of class names
    """
    # Get parameters from config if not specified
    if num_epochs is None:
        num_epochs = config.get('training.num_epochs', 10)
    if batch_size is None:
        batch_size = config.get('training.batch_size', 32)
    if learning_rate is None:
        learning_rate = config.get('training.learning_rate', 0.001)
    
    # Determine device (CPU or GPU)
    device_str = config.get_device()
    device = torch.device(device_str)
    
    # Enable CUDA optimizations if using GPU
    use_cuda = (device.type == 'cuda')
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        print("\n=== TRAINING ON GPU (CUDA) ===\n")
    else:
        print("\n=== TRAINING ON CPU ===\n")

    # Use preprocessed dataset for training
    preprocessed_dir = dataset_path.rstrip(os.sep) + '_preprocessed'
    if not os.path.exists(preprocessed_dir):
        print(f"[ERROR] Preprocessed dataset not found at {preprocessed_dir}. Please run preprocessing first.")
        return None, None
    
    print(f"[LOG] Using preprocessed dataset at {preprocessed_dir} for training")
    dataset_dir = preprocessed_dir

    # Initialize dataset with classes and transform
    classes = get_classes(dataset_dir)
    
    # Get transform with potential augmentation
    use_augmentation = config.get('training.use_augmentation', False)
    transform = TransformFactory.get_training_transform(use_augmentation)
    
    # Create full dataset
    full_dataset = FaceDataset(dataset_dir, classes, transform=transform)
    
    # Get sampling parameters
    known_sample_size = config.get('training.sample_per_class', 100)
    unknown_sample_size = config.get('training.sample_size', 15000)
    seed = config.get('training.seed', 42)
    random.seed(seed)
    
    # Sample dataset using manifest approach
    indices = sample_dataset(full_dataset, classes, known_sample_size, unknown_sample_size, seed)
    
    # Save training manifest
    save_manifest(full_dataset, indices, classes, dataset_dir)
    
    # Create data loader with sampled subset
    dataset = Subset(full_dataset, indices)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=use_cuda
    )

    # Initialize model
    model = create_model(classes, device)

    # Train model
    trained_model = train(model, train_loader, device, num_epochs, learning_rate)
    
    # Save model and classes
    save_model(trained_model, classes, model_save_path)
    classes_path = os.path.splitext(model_save_path)[0] + '.pkl'
    print(f"Model weights saved to: {model_save_path}")
    print(f"Class list saved to: {classes_path}")
    
    wait_key()
    return trained_model, classes

def sample_dataset(full_dataset, classes, known_sample_size, unknown_sample_size, seed):
    """
    Sample dataset for training, balancing known and unknown classes
    
    Args:
        full_dataset: Full dataset to sample from
        classes: List of class names
        known_sample_size: Maximum samples per known class
        unknown_sample_size: Maximum samples for unknown class
        seed: Random seed for sampling
        
    Returns:
        indices: List of indices to use for training
    """
    # Find label for 'unknown' class
    unknown_label = classes.index('unknown') if 'unknown' in classes else None
    
    # Sample from known classes
    known_indices = []
    for class_idx, class_name in enumerate(classes):
        if class_name == 'unknown':
            continue
        class_indices = [i for i, lbl in enumerate(full_dataset.labels) if lbl == class_idx]
        
        # Cap at requested sample size
        sample_size = min(known_sample_size, len(class_indices))
        if sample_size < len(class_indices):
            sampled = random.sample(class_indices, sample_size)
        else:
            sampled = class_indices
        known_indices.extend(sampled)
    
    # Sample from unknown class
    if unknown_label is not None:
        unknown_indices = [i for i, lbl in enumerate(full_dataset.labels) if lbl == unknown_label]
        sample_size = min(unknown_sample_size, len(unknown_indices))
        sampled_unknown = random.sample(unknown_indices, sample_size)
        indices = known_indices + sampled_unknown
        print(f"[LOG] Training on {len(known_indices)} known images (max {known_sample_size} per class) and {len(sampled_unknown)} sampled unknown images (seed={seed})")
    else:
        indices = known_indices
        print(f"[LOG] Training on {len(known_indices)} known images (max {known_sample_size} per class), no unknown class found (seed={seed})")
    
    return indices

def save_manifest(full_dataset, indices, classes, dataset_dir):
    """
    Save training manifest for reproducibility
    
    Args:
        full_dataset: Full dataset
        indices: Indices used for training
        classes: List of class names
        dataset_dir: Dataset directory to save manifest in
    """
    manifest = {}
    for idx in indices:
        cls_name = classes[full_dataset.labels[idx]]
        fname = os.path.basename(full_dataset.image_paths[idx])
        manifest.setdefault(cls_name, []).append(fname)
    
    manifest_path = os.path.join(dataset_dir, 'manifest.json')
    with open(manifest_path, 'w') as mf:
        json.dump(manifest, mf, indent=2)
    print(f"[LOG] Training manifest saved to: {manifest_path}")

def create_model(classes, device):
    """
    Create and initialize the model
    
    Args:
        classes: List of class names
        device: Torch device
        
    Returns:
        model: Initialized model on device
    """
    # Get input size from config
    target_size = config.get('image_processing.target_size', [128, 128])
    input_size = target_size[0] * target_size[1] * 3  # Width x Height x RGB
    
    # Create model with correct number of classes
    num_classes = len(classes)
    model = FaceRecognitionMLP(input_size, num_classes).to(device)
    
    return model

def train(model, train_loader, device, num_epochs, learning_rate):
    """
    Train model for the specified number of epochs
    
    Args:
        model: Model to train
        train_loader: DataLoader with training data
        device: Torch device
        num_epochs: Number of epochs to train for
        learning_rate: Learning rate for optimizer
        
    Returns:
        model: Trained model
    """
    # Setup loss function
    criterion = nn.CrossEntropyLoss()
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Setup early stopping if enabled
    early_stop_cfg = config.get('training.early_stopping', {})
    if early_stop_cfg.get('enabled', False):
        patience = early_stop_cfg.get('patience', 5)
        best_loss = float('inf')
        epochs_no_improve = 0
    else:
        patience = None
        best_loss = None
        epochs_no_improve = None
    
    # Training loop
    print_header("Training Face Recognition Model")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Iterate batches with progress bar
        for data, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
            # Move data and target to the training device with correct dtypes
            data = data.to(device=device, dtype=torch.float32)
            target = target.to(device=device, dtype=torch.long)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(data)
            loss = criterion(output, target)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update loss and accuracy
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        # Report epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
        
        # Early stopping check
        if patience is not None:
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered after epoch {epoch+1}")
                    break
    
    return model 