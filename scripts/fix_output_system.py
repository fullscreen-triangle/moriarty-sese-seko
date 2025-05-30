import os
import shutil
import argparse
import subprocess
import sys

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

def fix_dependencies(base_dir):
    """Fix Dask and other dependencies"""
    print("Fixing dependencies...")
    
    # Uninstall conflicting packages
    packages_to_remove = ['dask', 'distributed']
    for package in packages_to_remove:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', '-y', package])
            print(f"Uninstalled {package}")
        except subprocess.CalledProcessError:
            print(f"Package {package} was not installed")
    
    # Install correct dask version
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'dask[distributed]>=2023.9.0'])
        print("Installed dask[distributed]")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dask[distributed]: {e}")
        print("You may need to install manually: pip install dask[distributed]>=2023.9.0")

def main():
    parser = argparse.ArgumentParser(description='Fix Moriarty output system')
    parser.add_argument('--base-dir', default='.', help='Base directory of the project')
    parser.add_argument('--fix-deps', action='store_true', help='Fix dependency issues')
    args = parser.parse_args()
    
    # Create directory structure
    create_structure(args.base_dir)
    
    # Fix dependencies if requested
    if args.fix_deps:
        fix_dependencies(args.base_dir)
    
    print("Output system fix complete.")
    print("Next steps:")
    print("1. If you didn't use --fix-deps, run: pip uninstall dask distributed && pip install dask[distributed]>=2023.9.0")
    print("2. Run your analysis again to generate proper outputs.")
    print("3. The system will now gracefully fallback if Dask is not available.")

if __name__ == "__main__":
    main()
