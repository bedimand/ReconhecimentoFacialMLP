import os
import sys

def get_device():
    """
    Get the best available device (GPU or CPU)
    
    Returns:
        torch.device: The device to use
    """
    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

def clear_console():
    """Clear the console screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(title):
    """Print a formatted header"""
    width = min(os.get_terminal_size().columns, 80)
    print("=" * width)
    print(title.center(width))
    print("=" * width)

def wait_key():
    """Wait for a key press"""
    print("\nPress any key to continue...")
    if os.name == 'nt':  # Windows
        import msvcrt
        msvcrt.getch()
    else:  # Unix/Linux/Mac
        import termios
        import tty
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def create_person_dir(name, dataset_path="dataset"):
    """
    Create a directory for a person's face images
    
    Args:
        name: Person's name
        dataset_path: Base path for dataset
        
    Returns:
        str: Path to the person's directory
    """
    # Clean the name (remove special characters, replace spaces with underscores)
    clean_name = "".join(c if c.isalnum() else "_" for c in name).lower()
    
    # Create directory
    person_dir = os.path.join(dataset_path, clean_name)
    os.makedirs(person_dir, exist_ok=True)
    
    return person_dir

def count_images_by_person(dataset_path="dataset"):
    """
    Count how many images are available for each person
    
    Args:
        dataset_path: Path to dataset
        
    Returns:
        dict: Dictionary with person name as key and image count as value
    """
    counts = {}
    
    if not os.path.exists(dataset_path):
        return counts
    
    for person in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person)
        if not os.path.isdir(person_dir):
            continue
            
        # Count image files
        image_count = 0
        for file in os.listdir(person_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.pgm')):
                image_count += 1
                
        counts[person] = image_count
        
    return counts 