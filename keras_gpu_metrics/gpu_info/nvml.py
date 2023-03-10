from contextlib import contextmanager
from functools import lru_cache
import pynvml as nv
from pynvml import nvmlInit, nvmlShutdown
import time
from typing import List

from .gpu_status import GPUStatus


@contextmanager
def nvml_context():
    """Context manager for nvml.

    Credit: https://github.com/rossumai/nvgpu"""
    nvmlInit()
    yield
    nvmlShutdown()


@lru_cache(maxsize=1)
def _get_gpu_statuses(timestamp) -> List[GPUStatus]:
    """Returns a list of GPUStatus objects for each GPU on the system.

    Caches its results to avoid calling nvmlInit/nvmShutdown multiple times per batch.
    """
    with nvml_context():
        return [_get_gpu_status(i, timestamp) for i in range(_get_num_gpus())]


def get_gpu_statuses() -> List[GPUStatus]:
    """Returns a list of GPUStatus objects for each GPU on the system.

    It is safe to call this function repeatedly, but its return values will only update once per second.  GPU status
    results are cached using UNIX time rounded to the nearest second as the key.
    """
    return _get_gpu_statuses(int(time.time()))


def _get_gpu_status(device_index: int, timestamp: int) -> GPUStatus:
    """Gets the status of the specified GPU device.  Should be called within nvml_context().

    Returns (GPUStatus): The status of the GPU device.
    """
    handle = nv.nvmlDeviceGetHandleByIndex(device_index)
    device_name_data = nv.nvmlDeviceGetName(handle)
    try:
        device_name = device_name_data.decode('utf-8')
    except (UnicodeDecodeError, AttributeError):
        device_name = str(device_name_data)
    
    try:
        nv_procs = nv.nvmlDeviceGetComputeRunningProcesses(handle)
        pids = [p.pid for p in nv_procs]
    except nv.NVMLError:
        pids = None
    try:
        utilization = nv.nvmlDeviceGetUtilizationRates(handle).gpu
    except nv.NVMLError:
        utilization = None

    clock_speed_mhz = nv.nvmlDeviceGetClockInfo(handle, nv.NVML_CLOCK_SM)
    temperature = nv.nvmlDeviceGetTemperature(handle, nv.NVML_TEMPERATURE_GPU)
    memory = nv.nvmlDeviceGetMemoryInfo(handle)  # free, reserved, total, used
    try:
        fan_speed = nv.nvmlDeviceGetFanSpeed(handle)
    except nv.NVMLError:
        fan_speed = None
    power_usage_mw = nv.nvmlDeviceGetPowerUsage(handle)  # milliwatts
    return GPUStatus(
        timestamp=timestamp,
        gpu_id=device_index,
        device_name=device_name,
        pids=pids,
        utilization=utilization,
        clock_speed_mhz=clock_speed_mhz,
        temperature=temperature,
        memory_free=memory.free,
        memory_used=memory.used,
        fan_speed=fan_speed,
        power_usage_mw=power_usage_mw,
    )


def _get_num_gpus() -> int:
    """Returns the number of GPUs available on the system.  Should be called within nvml_context()."""
    return nv.nvmlDeviceGetCount()
