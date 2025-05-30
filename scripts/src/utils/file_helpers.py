import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def ensure_directory(path):
    """Ensure directory exists for file saving"""
    os.makedirs(os.path.dirname(path), exist_ok=True)

def save_image_safely(image, output_path, is_matplotlib=False):
    """Save image with proper error handling and directory creation"""
    try:
        ensure_directory(output_path)
        
        if is_matplotlib:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()  # Important to prevent memory leaks
        else:
            cv2.imwrite(output_path, image)
            
        # Verify file was written correctly
        if os.path.getsize(output_path) < 100:  # Suspiciously small file
            raise ValueError("Generated file is too small, likely corrupt")
            
        return True
    except Exception as e:
        print(f"Error saving image to {output_path}: {str(e)}")
        return False
