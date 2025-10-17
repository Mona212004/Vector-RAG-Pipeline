import torch
import multiprocessing

def get_device():
    if torch.backends.mps.is_available():
        num_mps = torch.mps.device_count()
        print(f"Detected {num_mps} MPS devices: {'Apple silicon GPU' if num_mps>0 else 'N/A'}")
        devices = ['mps'] * num_mps # Typically 1 device
    else: 
        print("No MPS detected; using CPU.")
        num_processes = multiprocessing.cpu_count()
        devices = ['cpu'] * num_processes
    return devices 
