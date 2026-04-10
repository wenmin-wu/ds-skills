---
name: timeseries-quaternion-angular-velocity
description: Derive angular velocity from consecutive quaternion frames via relative rotation and rotvec conversion
domain: timeseries
---

# Quaternion Angular Velocity

## Overview

When a sensor provides orientation as quaternions but not angular velocity directly, derive it from consecutive frames: compute the relative rotation between frames, convert to rotation vector, and divide by time delta. Useful as a feature for gesture/activity recognition.

## Quick Start

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

def angular_velocity_from_quats(quaternions, dt=1/200):
    """Derive angular velocity from consecutive quaternions.
    
    Args:
        quaternions: (N, 4) orientation [x, y, z, w]
        dt: time between samples (seconds)
    Returns:
        (N, 3) angular velocity [wx, wy, wz] in rad/s
    """
    n = len(quaternions)
    omega = np.zeros((n, 3))
    for i in range(n - 1):
        r_t = R.from_quat(quaternions[i])
        r_next = R.from_quat(quaternions[i + 1])
        delta = r_t.inv() * r_next
        omega[i] = delta.as_rotvec() / dt
    omega[-1] = omega[-2]  # fill last sample
    return omega
```

## Key Decisions

- **rotvec**: rotation vector magnitude = angle, direction = axis — dividing by dt gives angular velocity
- **dt from sampling rate**: 200 Hz IMU → dt = 0.005s
- **Angular distance**: `np.linalg.norm(delta.as_rotvec())` gives scalar rotation angle between frames

## References

- Source: [cmi25-imu-thm-tof-tf-blendingmodel-lb-82](https://www.kaggle.com/code/hideyukizushi/cmi25-imu-thm-tof-tf-blendingmodel-lb-82)
- Competition: CMI - Detect Behavior with Sensor Data
