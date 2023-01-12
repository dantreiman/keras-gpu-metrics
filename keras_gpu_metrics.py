"""Keras metrics to record GPU info during training or testing."""
from gpu_info.nvml import get_gpu_statuses
import tensorflow as tf


def _gpu_metric(gpu_index, property_getter, property_name):
    @tf.function
    def metric(y_true, y_logits):
        gpu_status = get_gpu_statuses()[gpu_index]
        return property_getter(gpu_status)
    metric.__name__ = f'gpu_{gpu_index}_{property_name}'
    return metric


def gpu_utilization_metric(gpu_index: int = 0):
    return _gpu_metric(gpu_index, lambda s: s.utilization, 'utilization')


def gpu_clock_speed_metric(gpu_index: int = 0):
    return _gpu_metric(gpu_index, lambda s: s.clock_mhz, 'clock_mhz')


def gpu_temperature_metric(gpu_index: int = 0):
    return _gpu_metric(gpu_index, lambda s: s.temperature, 'temperature')


def gpu_memory_free_metric(gpu_index: int = 0):
    return _gpu_metric(gpu_index, lambda s: s.memory_free, 'memory_free')


def gpu_memory_used_metric(gpu_index: int = 0):
    return _gpu_metric(gpu_index, lambda s: s.memory_used, 'memory_used')


def gpu_fan_speed_metric(gpu_index: int = 0):
    return _gpu_metric(gpu_index, lambda s: s.fan_speed, 'fan_speed')


def gpu_power_usage_metric(gpu_index: int = 0):
    return _gpu_metric(gpu_index, lambda s: s.power_usage, 'power_usage')
