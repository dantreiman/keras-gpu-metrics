from contextlib import contextmanager
from functools import lru_cache
import pynvml as nv
from pynvml import nvmlInit, nvmlShutdown
import time

from .gpu_status import GPUStatus


@contextmanager
def nvml_context():
    """Context manager for nvml.

    Credit: https://github.com/rossumai/nvgpu"""
    nvmlInit()
    yield
    nvmlShutdown()


@lru_cache(maxsize=3)
def _get_gpu_statuses(timestamp) -> list[GPUStatus]:
    """Returns a list of GPUStatus objects for each GPU on the system.

    Caches its results to avoid calling nvmlInit/nvmShutdown multiple times per batch.
    """
    with nvml_context():
        return [_get_gpu_status(i, timestamp) for i in range(_get_num_gpus())]


def get_gpu_statuses() -> list[GPUStatus]:
    """Returns a list of GPUStatus objects for each GPU on the system."""
    return _get_gpu_statuses(int(time.time()))


def _get_gpu_status(device_index: int, timestamp: int) -> GPUStatus:
    """Gets the status of the specified GPU device.  Should be called within nvml_context().

    Returns (GPUStatus): The status of the GPU device.
    """
    handle = nv.nvmlDeviceGetHandleByIndex(device_index)
    device_name = nv.nvmlDeviceGetName(handle).decode('UTF-8')
    nv_procs = nv.nvmlDeviceGetComputeRunningProcesses(handle)
    pids = [p.pid for p in nv_procs]
    try:
        utilization = nv.nvmlDeviceGetUtilizationRates(handle).gpu
    except nv.NVMLError:
        utilization = None

    clock_mhz = nv.nvmlDeviceGetClockInfo(handle, nv.NVML_CLOCK_SM)
    temperature = nv.nvmlDeviceGetTemperature(handle, nv.NVML_TEMPERATURE_GPU)
    memory = nv.nvmlDeviceGetMemoryInfo(handle)  # free, reserved, total, used
    fan_speed = nv.nvmlDeviceGetFanSpeed(handle)
    power_usage = nv.nvmlDeviceGetPowerUsage(handle)  # milliwatts
    return GPUStatus(
        timestamp=timestamp,
        gpu_id=device_index,
        device_name=device_name,
        pids=pids,
        utilization=utilization,
        clock_mhz=clock_mhz,
        temperature=temperature,
        memory_free=memory.free,
        memory_used=memory.used,
        fan_speed=fan_speed,
        power_usage=power_usage,
    )


def _get_num_gpus() -> int:
    """Returns the number of GPUs available on the system.  Should be called within nvml_context()."""
    return nv.nvmlDeviceGetCount()
