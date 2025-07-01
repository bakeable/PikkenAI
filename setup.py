#!/usr/bin/env python3
"""
Setup script for Pikken AI project.
"""

import subprocess
import sys
import os

def install_dependencies():
    """Install project dependencies."""
    print("Installing dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    dirs = [
        "models",
        "checkpoints", 
        "eval_logs",
        "tensorboard_logs"
    ]
    
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✓ Created directory: {dir_name}")

def main():
    """Main setup function."""
    print("=== Pikken AI Setup ===\n")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("✗ Python 3.8 or higher is required")
        return 1
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        return 1
    
    print("\n✓ Setup complete!")
    print("\nNext steps:")
    print("1. Run tests: python test_setup.py")
    print("2. Try a quick evaluation: python evaluate.py --games 50")
    print("3. Train an agent: python train.py --timesteps 10000")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
