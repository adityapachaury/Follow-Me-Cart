# Smart Follow-Me Logistics Cart

## Overview
The **Smart Follow-Me Logistics Cart** is a low-cost AI-powered motorized add-on kit designed for warehouse and industrial carts. Using computer vision, QR-based identity locking, and sensor-driven navigation, the cart autonomously follows a worker while maintaining a safe distance.

The system intelligently stops when the worker stops, avoids obstacles in real time, and searches for the worker if they disappear from the camera view. This solution improves warehouse efficiency, reduces manual effort, and enhances workplace safety.

---

# Key Features

## AI-Based Worker Tracking
- Detects and tracks workers using computer vision.
- Locks onto the assigned worker using a unique QR code on the worker’s vest.
- Prevents identity confusion in crowded warehouse environments.

## Autonomous Follow-Me Mode
- Automatically follows the assigned worker.
- Maintains a configurable safe distance.
- Provides smooth and stable movement control.

## Obstacle Detection & Avoidance
- Detects obstacles using sensors and vision models.
- Stops automatically when an obstacle blocks the path.
- Resumes movement once the path is clear.

## Smart Owner Re-Identification
- Stops when the worker disappears from the camera frame.
- Searches for the worker for up to 50 frames.
- Reconnects automatically if the worker is found.

## Manual Override
- Allows manual control in tight spaces.
- Includes emergency stop support for safety.

## Low-Cost Modular Design
- Designed as an add-on kit for existing carts.
- Uses affordable and easily available hardware components.

---

# System Workflow

## Step 1 — Owner Locking
1. Worker wears a QR-tagged vest.
2. Camera scans the QR code.
3. System locks the worker identity.
4. Cart begins following the worker.

## Step 2 — Autonomous Following
1. Computer vision continuously tracks the worker.
2. Cart maintains a safe following distance.
3. Motor controller dynamically adjusts movement.

## Step 3 — Obstacle Handling
1. Sensors detect obstacles in the path.
2. Cart immediately stops.
3. Movement resumes after the obstacle is removed.

## Step 4 — Owner Lost Detection
1. Worker leaves the camera frame.
2. Cart stops movement.
3. Searches for the worker for up to 50 frames.
4. Reconnects if found, otherwise enters idle mode.

---

# Technologies Used

| Category | Technologies |
|----------|-------------|
| Computer Vision | YOLOv8, OpenCV |
| Tracking | DeepSORT |
| AI Framework | PyTorch |
| Hardware Interface | Raspberry Pi / Arduino |
| Sensors | Ultrasonic / LiDAR |
| Programming Language | Python |
| Motor Control | PWM Motor Drivers |

---

# Hardware Requirements

- Raspberry Pi / NVIDIA Jetson Nano
- USB Camera / Pi Camera
- DC Motors with Motor Driver
- Ultrasonic or LiDAR Sensors
- Battery Pack
- QR-tagged Safety Vest
- Warehouse Cart

---

# Software Requirements

```bash
Python 3.10+
OpenCV
Ultralytics YOLO
deep-sort-realtime
PyTorch
NumPy
```
# Project Architecture
Camera Feed
     ↓
YOLOv8 Person Detection
     ↓
QR Identity Verification
     ↓
DeepSORT Tracking
     ↓
Distance Estimation
     ↓
Motor Control System
     ↓
Cart Movement



