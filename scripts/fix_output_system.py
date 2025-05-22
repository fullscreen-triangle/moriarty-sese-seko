import os
import shutil
import argparse

def create_structure(base_dir):
    """Create necessary directories and files for output system"""
    # Create required directories
    directories = [
        'src/utils',
        'src/templates',
    ]
    
    for directory in directories:
        os.makedirs(os.path.join(base_dir, directory), exist_ok=True)

def write_file(base_dir, relative_path, content):
    """Write content to file with proper directory creation"""
    full_path = os.path.join(base_dir, relative_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    
    with open(full_path, 'w') as f:
        f.write(content)
    
    print(f"Created file: {relative_path}")

def main():
    parser = argparse.ArgumentParser(description='Fix Moriarty output system')
    parser.add_argument('--base-dir', default='.', help='Base directory of the project')
    args = parser.parse_args()
    
    # Create directory structure
    create_structure(args.base_dir)
    
    # Write files (Implementation from functions defined above)
    # Example:
    file_helpers_content = """import os
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
"""
    write_file(args.base_dir, 'src/utils/file_helpers.py', file_helpers_content)
    
    # Write other files similarly...
    
    print("Output system fix complete. Run your analysis again to generate proper outputs.")

if __name__ == "__main__":
    main()
