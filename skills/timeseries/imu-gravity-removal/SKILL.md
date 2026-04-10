---
name: timeseries-imu-gravity-removal
description: Remove gravity component from raw accelerometer data using quaternion rotation to yield linear acceleration
domain: timeseries
---

# IMU Gravity Removal

## Overview

Raw accelerometer readings include gravitational acceleration (~9.81 m/s^2). Use the device's quaternion orientation to rotate the world-frame gravity vector into sensor frame, then subtract it. Essential preprocessing for any wearable/IMU activity recognition task.

## Quick Start

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

def remove_gravity(acc_xyz, quaternions, g=9.81):
    """Remove gravity from accelerometer using quaternion orientation.
    
    Args:
        acc_xyz: (N, 3) raw accelerometer [x, y, z]
        quaternions: (N, 4) orientation [x, y, z, w]
    Returns:
        (N, 3) linear acceleration
    """
    gravity_world = np.array([0, 0, g])
    linear = np.zeros_like(acc_xyz)
    for i in range(len(acc_xyz)):
        rot = R.from_quat(quaternions[i])
        gravity_sensor = rot.apply(gravity_world, inverse=True)
        linear[i] = acc_xyz[i] - gravity_sensor
    return linear
```

## Key Decisions

- **Quaternion format**: scipy uses `[x, y, z, w]` — check your sensor's convention
- **inverse=True**: transforms world→sensor frame (not sensor→world)
- **Vectorized option**: `R.from_quat(all_quats).apply(gravity, inverse=True)` for speed

## References

- Source: [cmi25-imu-thm-tof-tf-blendingmodel-lb-82](https://www.kaggle.com/code/hideyukizushi/cmi25-imu-thm-tof-tf-blendingmodel-lb-82)
- Competition: CMI - Detect Behavior with Sensor Data
