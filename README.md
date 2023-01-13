# keras-gpu-metrics

Keras callbacks and metrics for tracking GPU utilization, temperature, and power consumption.

Supports nvidia GPUs only through the pynvml library, a python wrapper for NVIDIA Management Library (NVML) APIs.

This library supports two main use cases:
- [Instantaneous metrics](#instantaneous-metrics): GPU utilization, temperature, and power consumption
- [Metric Tracking](#metric-tracking) during a training session

## Instantaneous Metrics

```python
from src.keras_gpu_metrics.gpu_info.nvml import get_gpu_statuses

# Gets a list of GPUStatus objects for each available GPU device
gpu_statuses = get_gpu_statuses()

# Prints the status of GPU 0
if len(gpu_statuses) > 0:
    print(gpu_statuses[0])
else:
    print('No compatible GPU detected')
```

Example output:
```
Status of GPU 0 at timestamp 1673587592 (f2023-01-12 21:26:32)
Device Name: NVIDIA GeForce RTX 3090
PIDs:        [1840, 1832, 9692]
Utilization: 2%
Clock Speed: 210 MHz
Temperature: 34 C
Memory Free: 891281408 bytes
Memory Used: 24878522368 bytes
Fan Speed:   0%
Power Usage: 21824 mW
```

## Metric Tracking

Metric tracking is supported using a Keras callback (GPUMetricTrackerCallback).  This callback should be added to the
list of callbacks passed to the `fit` method of a Keras model.  Metrics are updated on each batch and can be tracked
by using the metrics provided by the callback.

```python
from src.keras_gpu_metrics.keras_gpu_callbacks import GPUMetricTrackerCallback

# GPUMetricTrackerCallback is needed to update variables so that metrics (which are part of the tensorflow graph) can
# receive updated GPU info.
gpu_tracker_callback = GPUMetricTrackerCallback()

metrics = [
    tf.keras.metrics.BinaryAccuracy(),
    gpu_tracker_callback.utilization_metric(),
    gpu_tracker_callback.clock_speed_metric(),
    gpu_tracker_callback.temperature_metric(),
    gpu_tracker_callback.fan_speed_metric(),
    gpu_tracker_callback.power_usage_metric(),
]

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=metrics
)
```