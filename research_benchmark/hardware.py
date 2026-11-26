"""
research_benchmark.hardware
===========================
Hardware inspection, CUDA detection, resource profiling, and CPU safety guardrails.
"""

import os
import sys
import shutil
import psutil
import platform
import torch

def inspect_system_hardware() -> dict:
    """
    Performs comprehensive, safe inspection of local CPU, RAM, GPU, storage, and ML runtime.
    Does NOT assume CUDA or discrete GPU presence.
    """
    # CPU Profiling
    cpu_info = {
        "processor": platform.processor(),
        "machine": platform.machine(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_freq_current_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else None,
    }
    
    # RAM Profiling
    mem = psutil.virtual_memory()
    ram_info = {
        "total_gb": round(mem.total / (1024 ** 3), 2),
        "available_gb": round(mem.available / (1024 ** 3), 2),
        "used_gb": round(mem.used / (1024 ** 3), 2),
        "percent_used": mem.percent
    }
    
    # Storage Profiling
    cwd = os.getcwd()
    total_b, used_b, free_b = shutil.disk_usage(cwd)
    disk_info = {
        "drive": os.path.splitdrive(cwd)[0],
        "total_gb": round(total_b / (1024 ** 3), 2),
        "free_gb": round(free_b / (1024 ** 3), 2),
        "used_gb": round(used_b / (1024 ** 3), 2),
        "percent_used": round((used_b / total_b) * 100, 1)
    }
    
    # GPU / CUDA Profiling
    cuda_available = torch.cuda.is_available()
    gpu_info = {
        "cuda_available": cuda_available,
        "device_count": torch.cuda.device_count() if cuda_available else 0,
        "devices": []
    }
    if cuda_available:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free_vram, total_vram = torch.cuda.mem_get_info(i)
            gpu_info["devices"].append({
                "device_id": i,
                "name": props.name,
                "total_vram_gb": round(total_vram / (1024 ** 3), 2),
                "free_vram_gb": round(free_vram / (1024 ** 3), 2),
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count
            })
    else:
        gpu_info["fallback"] = "Host has no discrete NVIDIA CUDA GPU; execution restricted to CPU / ONNX Runtime."

    # ML Runtime Environment
    runtime_info = {
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "pytorch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "os_version": f"{platform.system()} {platform.release()} ({platform.version()})"
    }
    
    # Check optional packages
    for pkg in ["transformers", "datasets", "accelerate", "peft", "onnxruntime"]:
        try:
            mod = __import__(pkg)
            runtime_info[f"{pkg}_version"] = getattr(mod, "__version__", "installed")
        except ImportError:
            runtime_info[f"{pkg}_version"] = "NOT_INSTALLED"

    return {
        "cpu": cpu_info,
        "ram": ram_info,
        "disk": disk_info,
        "gpu": gpu_info,
        "runtime": runtime_info
    }

def configure_cpu_safety_limits(num_threads: int = 4):
    """
    Enforces safe CPU thread usage to avoid locking up host laptop during training.
    Leaves remaining cores for OS and user responsiveness.
    """
    torch.set_num_threads(num_threads)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(2)
        except RuntimeError:
            pass # Interop threads already initialized
    print(f"[*] CPU Safety Guard: PyTorch intra-op threads set to {num_threads}.")

def check_memory_headroom(min_free_mb: float = 500.0) -> bool:
    """
    Checks if host has sufficient free RAM before launching a training epoch.
    Returns False if available memory is below safety threshold.
    """
    mem = psutil.virtual_memory()
    free_mb = mem.available / (1024 * 1024)
    if free_mb < min_free_mb:
        print(f"[!] WARNING: Low memory! Only {free_mb:.1f} MB available (Minimum safe: {min_free_mb} MB).")
        return False
    return True

def get_current_process_memory_mb() -> float:
    """Returns the current process's physical memory footprint (RSS) in megabytes."""
    return psutil.Process().memory_info().rss / (1024 * 1024)

if __name__ == "__main__":
    profile = inspect_system_hardware()
    import pprint
    pprint.pprint(profile)
