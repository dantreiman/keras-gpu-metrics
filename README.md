# keras-gpu-metrics


Keras callbacks and metrics for tracking GPU utilization, temperature, and power consumption.

Supports nvidia GPUs only through the [nvidia-ml-py](https://pypi.org/project/nvidia-ml-py/) library,
a python wrapper for NVIDIA Management Library (NVML) APIs.

This library supports two main use cases:
- [Instantaneous metrics](#instantaneous-metrics): GPU utilization, temperature, and power consumption
- [Metric Tracking](#metric-tracking) during a training session

## Example notebooks:

### [GPU Info](gpu_info.ipynb)
[![GPU Info](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dantreiman/
keras-gpu-metrics/gpu_info.ipynb)
Basic example code to get current GPU status.

###[GPU Metrics](gpu_metrics_example.ipynb)
[![GPU Metrics](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dantreiman/
keras-gpu-metrics/gpu_metrics_example.ipynb)
Track GPU utilization, temperature, and power consumption during training of a tensorflow model.

### [Energy Usage](energy_usage_example.ipynb)
[![Energy Usage](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dantreiman/
keras-gpu-metrics/energy_usage_example.ipynb)

Demonstrates estimation of total GPU energy usage for training and testing a tensorflow model.

## APIs

### get_gpu_statuses()

```python
def get_gpu_statuses() -> List[GPUStatus]:
```
Returns a list of GPUStatus objects, one for each GPU on the system.

Usage:
```python
from keras_gpu_metrics import get_gpu_statuses

# Gets a list of GPUStatus objects for each available GPU device
gpu_statuses = get_gpu_statuses()
```

Returns:

`List[GPUStatus]`: A list of GPUStatus objects, one for each GPU on the system.

```python
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
```

### 


###

### 

## Instantaneous Metrics

```python
from keras_gpu_metrics.gpu_info import get_gpu_statuses

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

Tracking GPU stats as training metrics is supported using a Keras callback (GPUMetricTrackerCallback).
This callback should be added to the list of callbacks passed to the `fit` method of a Keras model.
Metric values are updated at the start of each batch and can be tracked by using the metric functions
provided by the callback object.

```python
from keras_gpu_metrics import GPUMetricTrackerCallback

# GPUMetricTrackerCallback is needed to update variables so that metrics
# (which are part of the tensorflow graph) can receive updated GPU info.
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

history = model.fit(
    dataset,
    epochs=10,
    validation_data=dataset,
    callbacks=[gpu_tracker_callback]
)
```