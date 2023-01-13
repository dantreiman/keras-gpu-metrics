from dataclasses import dataclass
from typing import List


@dataclass
class GPUStatus:
    timestamp: int        # UNIX timestamp (seconds) of this measurement
    gpu_id: int           # Integer ID of the GPU
    device_name: str      # Name of the GPU i.e. 'NVIDIA GeForce RTX 3090'
    pids: List[int]       # List of process IDs using the GPU
    utilization: int      # GPU utilization percentage
    clock_speed_mhz: int  # GPU clock speed in MHz
    temperature: int      # GPU temperature in Celsius
    memory_free: int      # Free GPU memory usage in bytes
    memory_used: int      # Used GPU memory usage in bytes
    fan_speed: int        # Fan speed percentage
    power_usage_mw: int   # Power usage in milliwatts
