from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


@dataclass
class GPUStatus:
    timestamp: int              # UNIX timestamp (seconds) of this measurement
    gpu_id: int                 # Integer ID of the GPU
    device_name: str            # Name of the GPU i.e. 'NVIDIA GeForce RTX 3090'
    pids: Optional[List[int]]   # List of process IDs using the GPU, or None if not supported.
    utilization: Optional[int]  # GPU utilization percentage, or None if not supported.
    clock_speed_mhz: int        # GPU clock speed in MHz
    temperature: int            # GPU temperature in Celsius
    memory_free: int            # Free GPU memory usage in bytes
    memory_used: int            # Used GPU memory usage in bytes
    fan_speed: Optional[int]    # Fan speed percentage, or None if not supported.
    power_usage_mw: int         # Power usage in milliwatts

    def __str__(self):
        dt_object = datetime.fromtimestamp(self.timestamp)
        return (
            f'Status of GPU {self.gpu_id} at timestamp {self.timestamp} ({str(dt_object)})\n'
            f'Device Name: {self.device_name}\n' +
            (f'PIDs:        {self.pids}\n' if self.pids is not None else 'PIDs:        unsupported\n') +
            (f'Utilization: {self.utilization}%\n' if self.utilization is not None else f'Utilization: unsupported\n') +
            f'Clock Speed: {self.clock_speed_mhz} MHz\n'
            f'Temperature: {self.temperature} C\n'
            f'Memory Free: {self.memory_free} bytes\n'
            f'Memory Used: {self.memory_used} bytes\n' +
            (f'Fan Speed:   {self.fan_speed}%\n' if self.fan_speed is not None else 'Fan Speed:   unsupported\n') +
            f'Power Usage: {self.power_usage_mw} mW'
        )
