import subprocess
import sys
import os

print("=== WhisperJAV Installation Debugger ===")
print(f"Python: {sys.executable}")
print(f"Installation dir: {sys.prefix}")

# Check pip
pip = os.path.join(sys.prefix, "Scripts", "pip.exe")
print(f"\nChecking pip at: {pip}")
print(f"Pip exists: {os.path.exists(pip)}")

if os.path.exists(pip):
    # List installed packages
    print("\nInstalled packages:")
    subprocess.run([pip, "list"])
    
    # Try to install a simple package
    print("\nTesting pip install...")
    subprocess.run([pip, "install", "--dry-run", "requests"])
    
    # Check git
    print("\nChecking git...")
    subprocess.run(["git", "--version"])

input("\nPress Enter to exit...")