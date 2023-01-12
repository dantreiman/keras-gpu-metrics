from dataclasses import dataclass, field
import logging
import numpy as np
import tensorflow as tf
import time
from typing import List

from gpu_info.nvml import get_gpu_statuses

logger = logging.getLogger(__name__)


class GPUMetricTrackerCallback(tf.keras.callbacks.Callback):
    """Creates tf.tensors representing GPU status."""
    def __init__(self, gpu_device=0):
        """
        :param gpu_device: Index (of list of indices) of GPU device to check temperature of.
        """
        self.gpu_device = gpu_device if isinstance(gpu_device, list) else [gpu_device]
        n_devices = len(self.gpu_device)
        self.utilization = tf.Variable(tf.zeros(n_devices), dtype=tf.float32, trainable=False)
        self.clock_speed = tf.Variable(tf.zeros(n_devices), dtype=tf.float32, trainable=False)
        self.temperature = tf.Variable(tf.zeros(n_devices), dtype=tf.float32, trainable=False)
        self.memory_free = tf.Variable(tf.zeros(n_devices), dtype=tf.float32, trainable=False)
        self.memory_used = tf.Variable(tf.zeros(n_devices), dtype=tf.float32, trainable=False)
        self.fan_speed = tf.Variable(tf.zeros(n_devices), dtype=tf.float32, trainable=False)
        self.power_usage = tf.Variable(tf.zeros(n_devices), dtype=tf.float32, trainable=False)

    def update_state_variables(self):
        """Checks GPU temperature, delays next batch if temp is too high."""
        device_status = [get_gpu_statuses()[i] for i in self.gpu_device]
        current_temperature = device_status.temperature
        self.temperatures.append(current_temperature)
        self.timestamps.append(device_status.timestamp)

    def on_train_batch_begin(self, batch, logs=None):
        self.check_temperature()

    def on_test_batch_begin(self, batch, logs=None):
        self.check_temperature()

    def on_predict_batch_begin(self, batch, logs=None):
        self.check_temperature()


class TemperatureCheckCallback(tf.keras.callbacks.Callback):
    """Checks GPU temp, delays next batch if temp is too high."""
    def __init__(self, gpu_device=0, max_allowed_temp=70, sleep_seconds=10):
        """
        :param gpu_device: Index of GPU device to check temperature of.
        :param max_allowed_temp: The maximum GPU temperature in Celsius.
        :param sleep_seconds: Seconds to delay the next batch if the GPU temperature is too high.
        """
        self.gpu_device = gpu_device
        self.max_allowed_temp = max_allowed_temp
        self.sleep_seconds = sleep_seconds
        self.temperatures = []
        self.timestamps = []

    def check_temperature(self):
        """Checks GPU temperature, delays next batch if temp is too high."""
        device_status = get_gpu_statuses()[self.gpu_device]
        current_temperature = device_status.temperature
        self.temperatures.append(current_temperature)
        self.timestamps.append(device_status.timestamp)
        if current_temperature > self.max_allowed_temp:
            logger.warning(f'GPU temperature {current_temperature}C exceeds max {self.max_allowed_temp}C.'
                           '  Delaying next batch.')
            time.sleep(self.sleep_seconds)

    def get_temperature_history(self):
        """After training, returns tuple of timestamps, temperatures"""
        return self.timestamps, self.temperatures

    def on_train_batch_end(self, batch, logs=None):
        self.check_temperature()

    def on_test_batch_end(self, batch, logs=None):
        self.check_temperature()

    def on_predict_batch_end(self, batch, logs=None):
        self.check_temperature()


@dataclass
class PowerUsageHistory:
    """Parallel arrays which store the power usage for each batch."""
    batch_index: List[int] = field(default_factory=list)
    batch_begin_time: List[float] = field(default_factory=list)
    batch_end_time: List[float] = field(default_factory=list)
    power_usage: List[int] = field(default_factory=list)

    def split_into_epochs(self):
        epoch_start_indices = tf.concat([
            tf.squeeze(tf.where([bi == 0 for bi in self.batch_index])),
            [-1]
        ], axis=0)
        return [
            PowerUsageHistory(
                batch_index=self.batch_index[epoch_start_indices[i]:epoch_start_indices[i + 1]],
                batch_begin_time=self.batch_begin_time[epoch_start_indices[i]:epoch_start_indices[i + 1]],
                batch_end_time=self.batch_end_time[epoch_start_indices[i]:epoch_start_indices[i + 1]],
                power_usage=self.power_usage[epoch_start_indices[i]:epoch_start_indices[i + 1]],
            ) for i in range(len(epoch_start_indices) - 1)
        ]

    def estimated_energy_use(self):
        """Returns estimated energy use in Wh."""
        dt = tf.math.subtract(self.batch_end_time, self.batch_begin_time)
        t = tf.cast(tf.math.cumsum(dt), dtype=tf.float32) * 1e-9  # Convert nanoseconds to seconds
        energy_mw_sec = np.trapz(y=self.power_usage, x=t)
        return energy_mw_sec / (3600.0 * 1000)  # convert to WH


class PowerMonitorCallback(tf.keras.callbacks.Callback):
    """Records power usage and batch elapsed time, approximates total power usage of a training run."""
    def __init__(self, gpu_device=0):
        """
        :param gpu_device: index of GPU device (or list of devices) to monitor power consumption.
        """
        self.gpu_devices = gpu_device if isinstance(gpu_device, list) else [gpu_device]
        self.train_power_usage = PowerUsageHistory()
        self.test_power_usage = PowerUsageHistory()
        self.predict_power_usage = PowerUsageHistory()

    def total_power(self):
        """Get instantaneous total GPU power usage."""
        device_statuses = get_gpu_statuses()
        return sum(device_statuses[i].power_usage for i in self.gpu_devices)

    def on_train_batch_begin(self, batch, logs=None):
        self.train_power_usage.batch_index.append(batch)
        self.train_power_usage.batch_begin_time.append(time.time_ns())

    def on_train_batch_end(self, batch, logs=None):
        self.train_power_usage.power_usage.append(self.total_power())
        self.train_power_usage.batch_end_time.append(time.time_ns())

    def on_test_batch_begin(self, batch, logs=None):
        self.test_power_usage.batch_index.append(batch)
        self.test_power_usage.batch_begin_time.append(time.time_ns())

    def on_test_batch_end(self, batch, logs=None):
        self.test_power_usage.power_usage.append(self.total_power())
        self.test_power_usage.batch_end_time.append(time.time_ns())

    def on_predict_batch_begin(self, batch, logs=None):
        self.predict_power_usage.batch_index.append(batch)
        self.predict_power_usage.batch_begin_time.append(time.time_ns())

    def on_predict_batch_end(self, batch, logs=None):
        self.predict_power_usage.power_usage.append(self.total_power())
        self.predict_power_usage.batch_end_time.append(time.time_ns())
