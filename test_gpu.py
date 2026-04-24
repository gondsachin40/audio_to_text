import os
import sys
import torch
from faster_whisper import WhisperModel

# --- ROBUST WINDOWS DLL FIX ---
def setup_cuda_paths():
    # Path to your site-packages in the venv
    venv_base = os.path.join(os.getcwd(), "env", "Lib", "site-packages")
    
    # Common locations for pip-installed nvidia DLLs on Windows
    # We add the 'bin' or 'lib' folders directly to the DLL search path
    potential_paths = [
        os.path.join(venv_base, "nvidia", "cublas", "bin"),
        os.path.join(venv_base, "nvidia", "cudnn", "bin"),
        os.path.join(venv_base, "nvidia", "cuda_runtime", "bin"),
    ]

    for path in potential_paths:
        if os.path.exists(path):
            print(f"Adding DLL directory: {path}")
            os.add_dll_directory(path)
            # Also add to system PATH for older compatibility
            os.environ["PATH"] = path + os.pathsep + os.environ["PATH"]

setup_cuda_paths()
# ------------------------------

print(f"CUDA Available in Torch: {torch.cuda.is_available()}")

try:
    # RTX 4060 loves float16
    model = WhisperModel("small", device="cuda", compute_type="float16")
    print("🚀 Success: Faster-Whisper loaded on GPU (RTX 4060)!")
except Exception as e:
    print(f"❌ GPU Load Failed: {e}")
    print("Trying CPU fallback...")
    model = WhisperModel("small", device="cpu", compute_type="int8")