import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from src.utils.config import config

def to_rgb(img):
    """Convert image to RGB if it isn't already"""
    return img.convert('RGB')

class TransformFactory:
    """
    Factory class for creating consistent transforms across the application
    All transformation creation is centralized here and pulls from config
    """
    
    @staticmethod
    def get_face_transform():
        """
        Get standard face transform pipeline used for both training and inference
        """
        # Pull values from config
        target_size = config.get('image_processing.target_size', [128, 128])
        norm_mean = config.get('image_processing.normalization.mean', [0.5, 0.5, 0.5])
        norm_std = config.get('image_processing.normalization.std', [0.5, 0.5, 0.5])
        
        # Expand single values to match RGB channels if needed
        if len(norm_mean) == 1:
            norm_mean = [norm_mean[0]] * 3
        if len(norm_std) == 1:
            norm_std = [norm_std[0]] * 3
            
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((target_size[1], target_size[0])),  # height, width
            transforms.Lambda(to_rgb),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std)
        ])
    
    @staticmethod
    def get_training_transform(use_augmentation=None):
        """
        Get transform for training, with optional augmentation
        """
        # Check config for augmentation setting if not explicitly passed
        if use_augmentation is None:
            use_augmentation = config.get('training.use_augmentation', False)
            
        if not use_augmentation:
            # Use standard transform if augmentation disabled
            return TransformFactory.get_face_transform()
        
        # Pull values from config
        target_size = config.get('image_processing.target_size', [128, 128])
        norm_mean = config.get('image_processing.normalization.mean', [0.5, 0.5, 0.5]) 
        norm_std = config.get('image_processing.normalization.std', [0.5, 0.5, 0.5])
        
        # Expand single values to match RGB channels if needed
        if len(norm_mean) == 1:
            norm_mean = [norm_mean[0]] * 3
        if len(norm_std) == 1:
            norm_std = [norm_std[0]] * 3
        
        # Augmentation transform with configurable parameters
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.Resize((target_size[1], target_size[0])),  # height, width
            transforms.Lambda(to_rgb),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std)
        ])
    
    @staticmethod
    def preprocess_face_tensor(face_img, device=None):
        """
        Preprocess a face image and return a tensor ready for model inference
        
        Args:
            face_img: The input face image (numpy array or PIL Image)
            device: Optional torch device to send tensor to
            
        Returns:
            face_tensor: Processed tensor with batch dimension added
        """
        transform = TransformFactory.get_face_transform()
        
        # Convert numpy array to PIL Image if needed
        if isinstance(face_img, np.ndarray):
            face_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        
        # Handle PIL Image input
        if isinstance(face_img, Image.Image):
            # Create a new transformation pipeline without ToPILImage
            target_size = config.get('image_processing.target_size', [128, 128])
            norm_mean = config.get('image_processing.normalization.mean', [0.5, 0.5, 0.5])
            norm_std = config.get('image_processing.normalization.std', [0.5, 0.5, 0.5])
            
            # Expand single values to match RGB channels if needed
            if len(norm_mean) == 1:
                norm_mean = [norm_mean[0]] * 3
            if len(norm_std) == 1:
                norm_std = [norm_std[0]] * 3
                
            # Create transform for PIL images directly
            pil_transform = transforms.Compose([
                transforms.Resize((target_size[1], target_size[0])),  # height, width
                transforms.Lambda(to_rgb),
                transforms.ToTensor(),
                transforms.Normalize(mean=norm_mean, std=norm_std)
            ])
            
            # Apply transform and add batch dimension
            face_tensor = pil_transform(face_img).unsqueeze(0)
        else:
            # Apply full transform and add batch dimension
            face_tensor = transform(face_img).unsqueeze(0)
        
        # Move to device if specified
        if device is not None:
            face_tensor = face_tensor.to(device)
            
        return face_tensor 

def recognize_face(model, face_img, classes, device):
    """
    Core face recognition function - given a face image, identify the person.
    
    Args:
        model: Face recognition model
        face_img: Face image (numpy array or PIL image)
        classes: List of class names
        device: Torch device to run inference on
        
    Returns:
        predicted_class: Name of predicted class
        confidence: Confidence score (0-100)
        all_probabilities: Dictionary mapping class names to probability scores
    """
    # Ensure we're in evaluation mode
    model.eval()
    
    # Set deterministic behavior
    with torch.no_grad():
        # Preprocess face image to tensor
        face_tensor = TransformFactory.preprocess_face_tensor(face_img, device)
        
        # Run inference
        outputs = model(face_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    
    # Get prediction and confidence
    predicted_idx = outputs.argmax(dim=1).item()
    predicted_class = classes[predicted_idx]
    confidence = probabilities[predicted_idx].item() * 100
    
    # Create dictionary of all probabilities
    all_probabilities = {
        classes[i]: probabilities[i].item() * 100 
        for i in range(len(classes))
    }
    
    return predicted_class, confidence, all_probabilities

