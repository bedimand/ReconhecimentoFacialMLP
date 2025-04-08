import torch
import torch.nn as nn
import os

class FaceMLP(nn.Module):
    """
    Multi-Layer Perceptron for facial recognition
    """
    def __init__(self, input_size=92*112, num_classes=1):
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

def load_model(model_path, device=None):
    """
    Load a trained model from disk
    
    Args:
        model_path: Path to the model file
        device: Torch device (CPU/GPU)
        
    Returns:
        model: Loaded model
        classes: List of class names
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    try:
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
            dataset_path = "dataset/"
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
        num_classes = len(classes)
        input_size = 92 * 112 * 1  # grayscale 92x112
        model = FaceMLP(input_size, num_classes)
        
        # Load weights
        model.load_state_dict(model_state)
        model.to(device)
        model.eval()  # Set to evaluation mode
        
        return model, classes
    
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, []

def save_model(model, classes, model_path):
    """
    Save model to disk
    
    Args:
        model: The model to save
        classes: List of class names
        model_path: Path to save the model
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': classes
    }, model_path)
    print(f"Model saved to: {model_path}") 