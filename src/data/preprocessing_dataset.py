import os
import cv2
import gc
import multiprocessing
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from src.face.preprocessing import PreprocessModule
from src.utils.config import config

# Multiprocessing setup for preprocessing
segmenter_pm = None

def _init_worker(target_size):
    """
    Initializer for each worker process: creates a PreprocessModule once.
    """
    global segmenter_pm
    segmenter_pm = PreprocessModule(target_size=target_size)

def _worker_task(args):
    """
    Worker task for preprocessing a single image
    """
    src_path, dst_path = args
    img = cv2.imread(src_path)
    if img is None or img.size == 0:
        return False
    # Remove background and resize using the per-process module
    proc = segmenter_pm.remove_background_grabcut(img)
    proc = cv2.resize(proc, segmenter_pm.target_size)
    cv2.imwrite(dst_path, proc)
    return True

def preprocess_dataset(original_dir, preprocessed_dir=None, target_size=None, sample_per_class=None):
    """
    Preprocess all images in original_dir with background removal and save to preprocessed_dir
    
    Args:
        original_dir: Source directory with raw images
        preprocessed_dir: Target directory for preprocessed images (default: original_dir + '_preprocessed')
        target_size: Target image size (default: from config)
        sample_per_class: Maximum samples per class (optional)
        
    Returns:
        success_count: Number of successfully preprocessed images
    """
    # Default preprocessed dir is original_dir + '_preprocessed'
    if preprocessed_dir is None:
        preprocessed_dir = original_dir.rstrip(os.sep) + '_preprocessed'
    
    # Get target size from config if not specified
    if target_size is None:
        target_size = tuple(config.get('image_processing.target_size', [128, 128]))
    
    print(f"[LOG] Preprocessing images from {original_dir} to {preprocessed_dir}")
    print(f"[LOG] Target size: {target_size}")
    
    os.makedirs(preprocessed_dir, exist_ok=True)
    
    # Collect preprocessing tasks
    tasks = []
    for cls in os.listdir(original_dir):
        src_folder = os.path.join(original_dir, cls)
        dst_folder = os.path.join(preprocessed_dir, cls)
        if not os.path.isdir(src_folder):
            continue
        
        # Create output folder
        os.makedirs(dst_folder, exist_ok=True)
        
        # Get list of files to process
        files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
        
        # Sample if requested
        if sample_per_class and len(files) > sample_per_class:
            import random
            files = random.sample(files, sample_per_class)
            print(f"[LOG] Sampling {sample_per_class} images from {cls} class")
        
        # Add tasks for this class
        for fname in files:
            src_path = os.path.join(src_folder, fname)
            dst_path = os.path.join(dst_folder, fname)
            tasks.append((src_path, dst_path))
    
    # Process with processes and progress bar, track successes and failures
    success_count = 0
    failure_count = 0
    
    if tasks:
        # Use one process per CPU
        with ProcessPoolExecutor(
            max_workers=multiprocessing.cpu_count(), 
            initializer=_init_worker, 
            initargs=(target_size,)
        ) as executor:
            for ok in tqdm(executor.map(_worker_task, tasks), total=len(tasks), desc="Preprocessing images", unit="img"):
                if ok:
                    success_count += 1
                else:
                    failure_count += 1
    
    print(f"[LOG] Preprocessed {success_count}/{len(tasks)} images, {failure_count} failures.")
    
    # Clean up per-process module
    global segmenter_pm
    segmenter_pm = None
    gc.collect()
    
    return success_count 