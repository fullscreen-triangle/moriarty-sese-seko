#!/usr/bin/env python3
"""
Quick runner for Moriarty Framework Showcase
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = ['numpy', 'matplotlib', 'seaborn']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\n💡 Install with: pip install numpy matplotlib seaborn")
        return False
    
    return True

def main():
    print("🚀 Moriarty Framework Showcase Runner")
    print("="*50)
    
    # Check if models directory exists
    if not os.path.exists("models"):
        print("❌ Models directory not found!")
        print("   Please ensure the 'models' folder exists with JSON files.")
        return
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Run the main showcase
    print("✅ Dependencies check passed")
    print("🎯 Starting Moriarty Framework showcase...\n")
    
    try:
        # Import and run the showcase
        import showcase_moriarty
        showcase_moriarty.main()
        
    except Exception as e:
        print(f"❌ Error running showcase: {str(e)}")
        print("\n💡 Try running directly: python showcase_moriarty.py")

if __name__ == "__main__":
    main() 