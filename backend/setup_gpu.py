"""
GPU Acceleration Setup Script for NVIDIA RTX 4050
Installs GPU-accelerated Qiskit packages
"""

import subprocess
import sys

def run_command(cmd):
    """Run a command and print output"""
    print(f"\n{'='*60}")
    print(f"Running: {cmd}")
    print('='*60)
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(e.stderr)
        return False

def main():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║  GPU Acceleration Setup for Quantum Simulation          ║
    ║  NVIDIA RTX 4050                                         ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    print("\n📋 Prerequisites:")
    print("   1. CUDA Toolkit 12.x installed")
    print("   2. Latest NVIDIA drivers")
    print("   3. Python 3.8+")
    
    input("\nPress Enter to continue with installation...")
    
    # Install GPU-accelerated packages
    packages = [
        "qiskit-aer-gpu",
        "cupy-cuda12x",
    ]
    
    print("\n🚀 Installing GPU-accelerated packages...")
    
    for package in packages:
        success = run_command(f"pip install {package}")
        if not success:
            print(f"\n⚠️  Failed to install {package}")
            print("   This is optional - CPU version will still work")
    
    # Verify installation
    print("\n\n🔍 Verifying GPU availability...")
    
    verify_code = """
from qiskit_aer import AerSimulator
devices = AerSimulator.available_devices()
print(f'Available devices: {devices}')
if 'GPU' in devices:
    print('✅ GPU acceleration is ENABLED!')
else:
    print('⚠️  GPU not detected. Using CPU.')
"""
    
    with open('verify_gpu.py', 'w') as f:
        f.write(verify_code)
    
    run_command("python verify_gpu.py")
    
    print("\n\n" + "="*60)
    print("✅ Setup Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Restart your backend server")
    print("2. Check logs for '🚀 Executing quantum circuit on GPU'")
    print("3. Monitor GPU usage with: nvidia-smi -l 1")
    print("\n")

if __name__ == "__main__":
    main()
