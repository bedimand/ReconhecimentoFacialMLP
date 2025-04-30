import os
import cv2
import torch
from torch.utils.data import Dataset

class FaceDataset(torch.utils.data.Dataset):
    """
    Dataset for face recognition with preprocessed images
    """
    def __init__(self, dataset_path, classes, transform=None, is_train=True):
        """
        Args:
            dataset_path: Path to dataset folder.
            classes: List of class names (folders).
            transform: Optional transform to apply to images.
            is_train: If True, use images for training; if False, use for testing.
        """
        self.image_paths = []
        self.labels = []
        self.transform = transform

        for label, person_name in enumerate(classes):
            person_folder = os.path.join(dataset_path, person_name)
            if os.path.isdir(person_folder):
                for img_name in os.listdir(person_folder):
                    img_path = os.path.join(person_folder, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return image, label


class BackgroundRemovalDataset(Dataset):
    """
    Dataset for loading and processing images with background removal
    """
    def __init__(self, dataset_path, preprocess_module):
        """
        Args:
            dataset_path: Path to the dataset folder.
            preprocess_module: PreprocessModule for background removal, resizing, etc.
        """
        self.dataset_path = dataset_path
        self.preprocess_module = preprocess_module
        self.image_paths = []
        self.labels = []
        
        # Load image paths and labels
        for label, person_name in enumerate(os.listdir(dataset_path)):
            person_folder = os.path.join(dataset_path, person_name)
            if os.path.isdir(person_folder):
                for img_name in os.listdir(person_folder):
                    self.image_paths.append(os.path.join(person_folder, img_name))
                    self.labels.append(label)

    def __len__(self):
        """Return the size of the dataset"""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Get an image and its label from the dataset"""
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load the image
        image = cv2.imread(image_path)

        # Preprocess the image (background removal, resizing, and normalization)
        processed_image = self.preprocess_module.preprocess_face(image)

        return processed_image, label

def get_classes(dataset_path):
    """
    Return a sorted list of class (person) names from the dataset directory.
    
    Args:
        dataset_path: Path to the dataset folder
        
    Returns:
        List of class names (sorted alphabetically)
    """
    if not os.path.exists(dataset_path):
        return []
    classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    classes.sort()
    return classes 