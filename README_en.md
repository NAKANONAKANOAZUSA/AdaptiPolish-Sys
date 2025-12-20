# AdaptiPolish-Sys

**AdaptiPolish-Sys** is an automated **pre-polishing and defect-detection system** based on **point cloud processing and deep learning**. By integrating **robot arm control**, **computer vision**, and **real-time monitoring**, the system enables fully automated grinding and polishing workflows.

The system supports **workpiece contour recognition**, **path planning**, and **grinding parameter configuration**, and is suitable for surface finishing of **3D-printed parts**, **wood**, and similar materials.

## 1. Installation Guide

### 1.1 One-Click Installation

```bash
cd MasterComputer
```

### 1.2 Create a Virtual Environment

```bash
# Windows
python -m venv ap_env
ap_env\Scripts\activate

# Linux
python3 -m venv ap_env
source ap_env/bin/activate
```

### 1.3 Install Master Computer Dependencies

**Python version: 3.12**

```bash
pip install -r requirements-MasterComputer.txt
```

### 1.4 Install Camera Dependencies

**Python version: 3.9**

```bash
pip install -r requirements-Camera.txt
```

### 1.5 Use Pre-Packaged Virtual Environment (Optional)

You may directly use the packaged virtual environment.
**Requirement:** same CPU architecture and OS (**Bianbu**).

### 1.6 Platform-Specific Dependencies

```bash
# Ubuntu
sudo apt-get install portaudio19-dev libopenblas-dev
```

## 2. Robot Arm Connection

Connect the **MyCobot 280** robot arm to the computer via **serial (USB)** or **Ethernet**.

## 3. Camera Service Configuration

1. Connect the camera via **Ethernet**.
2. Open `CameraService.py` and modify `python_path` to point to the actual **Python 3.9** executable path.
3. Start the **CameraService**.

## 4. Point Cloud (C++) Deployment

### 4.1 Set up the project directory structure

Example:

```text
C:\pointcloud_project\
├── data\
│   └── train\        # Training data
└── src\              # Runtime files
    ├── Dataset.py
    ├── build_dataloader.py   # GPU version
    └── build_dataloader2.py  # CPU version
```

### 4.2 Required Environment (Point Cloud)

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy scikit-learn tqdm
```

### 4.3 Build C++ Point Cloud Module

1. Create `pc_backend.cpp`

2. Create `CMakeLists.txt`

3. Run the following commands:

   ```bash
   cd ~/Desktop/cse/pointcloud_project/src   # Custom path
   rm -rf build
   mkdir build
   cd build
   cmake ..
   make -j4
   ```

   After a successful build:
   - A shared library (`.so`) is generated
   - The library can be loaded directly in Python
   - Python can now use the C++ point cloud backend for high-performance processing

### 4.4 Start the System

```bash
python main.py
```

## 5. User Interface Layout Overview

The main interface is divided into three sections:

1. **Left Panel**
   Live camera feed and detection controls

2. **Center Panel**
   Teaching point management, grinding control, and motion control

3. **Right Panel**
   Camera configuration, robot connection, and auxiliary tools

<img width="1920" height="1048" alt="主界面" src="https://github.com/user-attachments/assets/63966602-38b9-49b0-868b-7293d5ab536e" />

## 6. Robot Connection

1. In the **Robot Connection** panel on the right:

   - Enter the IP address (default: `192.168.25.181`)
   - Enter the port (default: `9000`)
2. Click **“Connect Robot”**

<img width="1920" height="1048" alt="链接机器人" src="https://github.com/user-attachments/assets/2d85b542-21cf-4d12-8ce4-bc2b110019f2" />

After a successful connection, the status indicator turns **green**.

> **Note:** Ensure the robot arm is powered on and connected via USB before first use.

## 7. Camera Setup and Calibration

### 7.1 Camera Configuration

In the **Camera Settings** panel on the right:

- Select **camera type** (Local / Network)
- Enter **IP** and **port** for network cameras (if using a network camera)
- Load the ONNX model file with the button **Load**

### 7.2 Calibration

1. Click **“Start Calibration”**
2. Place **two ArUco markers** in the camera’s field of view
   (distance between markers: **220 mm**)
3. The system automatically completes calibration and displays **“Calibration Successful”**

<img width="1920" height="1048" alt="连接摄像头" src="https://github.com/user-attachments/assets/69c08048-1c96-43de-8126-8f4af3272267" />

### 7.3 Point Cloud Acquisition

1. Click **“Acquire Point Cloud Data”**
2. The system triggers the 3D camera to scan the object
3. The scanned point cloud is compared against the reference model
4. Defect point coordinates are returned automatically

<img width="1920" height="1048" alt="点云设置" src="https://github.com/user-attachments/assets/01e132a6-6b88-46b7-99a1-27ea55345817" />

> The 3D camera calibration matrix is built into the system.
> Returned defect coordinates are automatically transformed into the robot coordinate system.

## 8. Teaching Point Management

1. Move the robot arm to the desired position
2. In the **Teaching Point Management** panel at middle:
   - Enter a point name
   - Click **“Save Current Position”**

In the teaching point list:

- Select a point and click **“Move to Selected Point”** to move to the selected point
- Click **“Delete Selected Point”** to remove the unwanted point

<img width="1920" height="1048" alt="视角点管理" src="https://github.com/user-attachments/assets/532f1e87-2395-4b54-86df-fd13d34d708a" />

## 9. Contour Detection and Grinding

1. In the Left Panel, click **“Start Detection”**
2. Place the workpiece within the camera’s field of view
3. The system automatically detects and displays the workpiece contour

In the middle panel **Grinding Control**:

- Set coordinate offsets (X / Y / Z)
- Configure grinding parameters:
  - Number of cycles
  - Depth per pass
  - Scaling factor
- Click **“Apply Parameters”**

Click **“Start Grinding”** to begin the automated polishing process.

## 10. Manual Control

1. Click **“Manual Control”** in the right panel

<img width="1920" height="1048" alt="启动手动控制" src="https://github.com/user-attachments/assets/f69a309c-a52e-4a3d-84a5-a30c17f17636" />

In the dialog:

- Enable / release individual joints
- Enable or disable free-movement mode
- Save the current position as a teaching point

<img width="506" height="728" alt="手动控制" src="https://github.com/user-attachments/assets/6565b31b-2f94-4449-a6f4-e0c0f5cafe22" />

## 11. Motor Control

Click **“Motor Control”** in the right panel

<img width="1920" height="1048" alt="启动电机控制" src="https://github.com/user-attachments/assets/90451a36-3d50-4c38-b163-83b6f0d35460" />

In the dialog:

- Button **“Connect Motor”** for initial motor connection
- Control motor start/stop and forward/reverse rotation
- Adjust motor speed using the slider (range: **0–500**)

<img width="406" height="466" alt="电机控制" src="https://github.com/user-attachments/assets/63aa3040-0fd1-47c0-8ddf-676b6aa2c121" />

## 12. Coordinate Movement

1. In the middle panel **Execution Control**, locate the section **XYZ Coordinate Movement**
2. Enter target X / Y / Z coordinates
3. Enable **Angle Compensation** (recommended)
4. Click **“Move”** to execute positioning

<img width="1920" height="1048" alt="运动控制" src="https://github.com/user-attachments/assets/ef823558-c823-44eb-92f1-08054d771ac2" />

## 13. Advanced Features

**Cyclic Path Grinding**
1. Set **cycle count > 1** in grinding parameters
2. The system performs **progressive, multi-layer grinding automatically**

<img width="1920" height="1048" alt="打磨控制" src="https://github.com/user-attachments/assets/cbc41cdb-6ee4-47d2-a9f7-7743196dc7ca" />

<img width="1920" height="1048" alt="执行控制" src="https://github.com/user-attachments/assets/0c617277-f5eb-4877-a517-f58d6dbe0455" />

## 14. Hardware and Wiring

| Hardware Component        | Notes                           |
| ------------------------- | ------------------------------- |
| R-3650 BLDC Motor         | Motor                           |
| BLDC PWM Speed Controller | RS-485 or USB communication     |
| RS-485 to USB Converter   | —                               |
| Switching Power Supply    | 220V → 24V 5A                   |
| USB Microphone            | Plug-and-play                   |
| USB Speaker               | Plug-and-play                   |
| USB Hub                   | Multi-port, USB 3.0 recommended |
| Chaofeng Pro 3D Camera    | 3D vision camera                |

<img width="500" height="600" alt="image" src="https://github.com/user-attachments/assets/d4ddd8a0-7b44-4abf-83a5-4650702eff08" />

## 15. Virtual Environment Package

Due to repository upload limitations, the virtual environment is provided via cloud storage:

**File:** `venvMasterComputer.7z`
**Link:** [https://pan.baidu.com/s/1NTiiQDvPTQGQCAXc88vbuA](https://pan.baidu.com/s/1NTiiQDvPTQGQCAXc88vbuA)
**Access Code:** `1234`

