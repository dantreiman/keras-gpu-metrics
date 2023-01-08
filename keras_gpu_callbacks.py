import logging
import tensorflow as tf
import time

from gpu_info.nvml import get_gpu_statuses

logger = logging.getLogger(__name__)


class TemperatureCheck(tf.keras.callbacks.Callback):
    """Checks GPU temp, delays next batch if temp is too high."""
    def __init__(self, gpu_device=0, max_temp=70, sleep_seconds=10):
        """
        :param gpu_device: GPU device number to check temperature of.
        :param max_temp: The maximum GPU temperature in Celsius.
        :param sleep_seconds: Seconds to delay the next batch if the GPU temperature is too high.
        """
        self.gpu_device = gpu_device
        self.max_temp = max_temp
        self.sleep_seconds = sleep_seconds

    def check_temperature(self):
        """Checks GPU temperature, delays next batch if temp is too high."""
        device_status = get_gpu_statuses()[self.gpu_device]
        current_temperature = device_status.temperature
        if current_temperature > self.max_temp:
            logger.warning(f'GPU temperature {current_temperature}C exceeds max {self.max_temp}C.  Delaying next batch.')
            time.sleep(self.sleep_seconds)

    def on_train_batch_end(self, batch, logs=None):
        self.check_temperature()

    def on_test_batch_end(self, batch, logs=None):
        self.check_temperature()

    def on_predict_batch_end(self, batch, logs=None):
        self.check_temperature()
