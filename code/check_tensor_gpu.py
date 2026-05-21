import os
import sys

# 1. Sneakily find the site-packages path to symlink BEFORE importing tensorflow
try:
    import torch
    print("--- Torch (SBERT Engine) ---")
    print(f"Available: {torch.cuda.is_available()}")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
except ImportError:
    print("Torch not installed yet.")

# 2. Run the directory fix using a generic site-packages look up
original_dir = os.getcwd()
for path in sys.path:
    target_dir = os.path.join(path, 'tensorflow')
    if os.path.isdir(target_dir):
        os.chdir(target_dir)
        os.system('ln -sf ../nvidia/*/lib/*.so* .')
        os.chdir(original_dir)
        break

# 3. NOW it is safe to import TensorFlow and scan for hardware
import tensorflow as tf

print("\n--- TensorFlow (KerasMLP Engine) ---")
gpu_devices = tf.config.list_physical_devices('GPU')
print(f"Available: {len(gpu_devices) > 0}")
if gpu_devices:
    print(f"Device: {gpu_devices[0]}")