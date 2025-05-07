import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os
from src.utils.config import config

class FaceRecognitionMLP(nn.Module):
    def __init__(self, input_size=None, num_classes=None):
        """
        Define the layers of the MLP for face recognition.

        Args:
        - input_size (int): The number of input features (width * height * channels).
            If None, pulled from config.
        - num_classes (int): The number of classes (people + "unknown").
            If None, caller should provide this based on the dataset.
        """
        super(FaceRecognitionMLP, self).__init__()
        
        # Get input size from config if not provided
        if input_size is None:
            target_size = config.get('image_processing.target_size', [128, 128])
            input_size = target_size[0] * target_size[1] * 3  # Width x Height x RGB
        
        # Get dropout rate from config
        dropout_rate = config.get('training.dropout_rate', 0.0)
        
        # Define the layers of the MLP
        self.fc1 = nn.Linear(input_size, 2048)  # First fully connected layer
        self.fc2 = nn.Linear(2048, 1024)        # Second fully connected layer
        self.fc3 = nn.Linear(1024, 512)         # Third fully connected layer
        self.fc4 = nn.Linear(512, 128)          # Fourth fully connected layer
        self.fc5 = nn.Linear(128, num_classes)  # Output layer based on number of classes
        
        # Add dropout if configured
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
        - x (torch.Tensor): The input image tensor (after preprocessing).
        
        Returns:
        - output (torch.Tensor): The output prediction of the model.
        """
        # Flatten the input (batch_size, C, H, W) -> (batch_size, C*H*W)
        x = x.view(x.size(0), -1)
        
        # Pass through the layers with ReLU activations
        x = F.relu(self.fc1(x))  # Apply ReLU after the first layer
        
        # Apply dropout if configured
        if self.dropout is not None:
            x = self.dropout(x)
            
        x = F.relu(self.fc2(x))  # Apply ReLU after the second layer
        
        if self.dropout is not None:
            x = self.dropout(x)
            
        x = F.relu(self.fc3(x))  # Apply ReLU after the third layer
        
        if self.dropout is not None:
            x = self.dropout(x)
            
        x = F.relu(self.fc4(x))  # Apply ReLU after the fourth layer
        
        # Final output layer (no activation here, raw logits for classification)
        output = self.fc5(x)
        
        return output


def save_model(model, classes, model_path, classes_path=None):
    """
    Save the model weights and class list.
    Args:
        model: Trained FaceRecognitionMLP model
        classes: List of class names
        model_path: Path to save model weights (.pth)
        classes_path: Path to save class list (.pkl). If None, uses model_path with .pkl extension.
    """
    # Ensure model directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    
    if classes_path is None:
        classes_path = os.path.splitext(model_path)[0] + '.pkl'
    # Ensure classes directory exists
    os.makedirs(os.path.dirname(classes_path), exist_ok=True)
    with open(classes_path, 'wb') as f:
        pickle.dump(classes, f)


def load_model(model_path, device, classes_path=None):
    """
    Load the model weights and class list.
    Args:
        model_path: Path to model weights (.pth)
        device: torch.device
        classes_path: Path to class list (.pkl). If None, uses model_path with .pkl extension.
    Returns:
        model: Loaded FaceRecognitionMLP model
        classes: List of class names
    """
    # Set random seed for consistent initialization
    seed = config.get('training.seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    if classes_path is None:
        classes_path = os.path.splitext(model_path)[0] + '.pkl'
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found")
        return None, None
    
    if not os.path.exists(classes_path):
        print(f"Error: Classes file {classes_path} not found")
        return None, None
    
    # Load class list
    try:
        with open(classes_path, 'rb') as f:
            classes = pickle.load(f)
        num_classes = len(classes)
    except Exception as e:
        print(f"Error loading classes: {e}")
        return None, None
    
    # Get input size from config
    target_size = config.get('image_processing.target_size', [128, 128])
    input_size = target_size[0] * target_size[1] * 3  # Width x Height x RGB
    
    # Create model with the right input and output sizes
    model = FaceRecognitionMLP(input_size, num_classes)
    
    # Load state dict with strict=True to ensure all parameters are loaded correctly
    try:
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model parameters: {e}")
        print("This could cause inconsistent recognition results.")
        return None, None
    
    model.to(device)
    model.eval()
    return model, classes
