#  FIGHT INCIDENT DETECTION IN CCTV FOOTAGE USING DEEP LEARNING MODELS

A real-time fight/violence detection system for CCTV footage, powered by a hybrid deep learning pipeline (**YOLOv8-Pose + EfficientNetB0 + LSTM**) and served through a Flask web dashboard.

Developed by **Adam Ishraq Bin Addie Irwan** as a Final Year Project, Faculty of Information Science and Technology (FTSM), Universiti Kebangsaan Malaysia (UKM).
Supervisor: Assoc. Prof. Ts. Dr. Nor Samsiah Sani

---

##  Overview

Manual CCTV monitoring is limited by human attention span — operator focus typically drops after just 8–10 minutes of continuous viewing, causing violent incidents to go undetected. This system automates that detection by combining:

- **Human pose estimation** (YOLOv8-Pose) — extracts 17 body keypoints per person
- **Spatial feature extraction** (EfficientNetB0 CNN) — captures visual context beyond skeleton data
- **Temporal sequence modelling** (LSTM) — recognizes fight patterns across a sequence of frames

When a fight is detected, the system displays a red bounding box with a confidence score on the video feed and logs the event in real time on the dashboard.

##  Features

-  Upload and analyze CCTV video footage frame-by-frame
-  Real-time bounding box + confidence score overlay on detected individuals
-  Live activity log with timestamps
-  Dark mode dashboard UI
-  Robust error handling (unsupported formats, corrupted files, oversized uploads)

##  Architecture

```
CCTV Video → YOLOv8-Pose (keypoints) → EfficientNetB0 (spatial features) → LSTM (temporal modelling) → Fight / Normal Classification → Flask Dashboard
```

Three architectures were developed and benchmarked before settling on the final model:

| Model | Architecture | Accuracy |
|---|---|---|
| Model A | YOLOv8-Pose + LSTM | 84.73% |
| Model B | YOLOv8-Pose + MobileNetV2 + LSTM | 92.70% |
| **Model C** ⭐ | **YOLOv8-Pose + EfficientNetB0 + LSTM** | **96.64%** |

After hyperparameter tuning with **Optuna** (Tree-structured Parzen Estimator + Hyperband pruning, 30 trials), Model C's performance improved further:

| Metric | Before Tuning | After Tuning |
|---|---|---|
| Accuracy | 96.64% | **99.32%** |
| Precision | 96% | 99% |
| Recall | 97% | **100%** |
| F1-Score | 97% | 99% |

A 100% recall means the model detected every fight incident in the test set with zero false negatives — critical for a security-alerting system.

##  Dataset

- 500 videos total (250 fight, 250 normal), sourced from the **Surveillance Fight Dataset** and **Peliculas Dataset**
- Split 80:10:10 (train/val/test) using Video-Grouped Splitting to prevent data leakage
- Person tracking across frames handled by **ByteTrack**

##  Testing & Validation

- **Black-box testing**: 6/6 functional scenarios passed (normal video, fight video, unsupported file formats, corrupted files, empty submission, oversized files)
- **User Acceptance Test (UAT)**: 13 respondents (security guards, security management, technical staff) — every evaluation item scored above 4.0/5.0 on a 5-point Likert scale
  

## 🛠️ Tech Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript
- **Deep Learning**: PyTorch, Ultralytics YOLOv8-Pose, ONNX Runtime
- **Computer Vision**: OpenCV
- **Hyperparameter Tuning**: Optuna

##  Prerequisites

- Python 3.8 or newer
- Internet connection (for installing dependencies on first run)

##  Installation & Usage (Windows)

1. **Extract the project files**
   Extract `source code.zip` to a location on your computer (e.g. Desktop or Downloads).
   Source code.zip is in Releases section v.1.0 (FYP source code cleaned.zip).

3. **Open Command Prompt inside the `dashboard` folder**
   - Open the extracted `source code` folder, then open the `dashboard` folder inside it.
   - Click the address bar at the top of File Explorer.
   - Clear the path text, type `cmd`, and press Enter.
   - A Command Prompt window will open directly in the `dashboard` folder.

4. **Install dependencies**
   Run the following command:
   ```bash
   pip install flask werkzeug opencv-python torch torchvision onnxruntime numpy ultralytics
   ```
   *This may take a few minutes depending on your internet speed.*

5. **Start the server**
   ```bash
   python app.py
   ```
   or
   ```bash
   py app.py
   ```
   Wait until you see: `Running on http://127.0.0.1:5000`

6. **Open the dashboard**
   Open your browser (Chrome / Edge / Firefox) and go to:
   ```
   http://127.0.0.1:5000
   ```

##  Testing the System

Sample test videos are included in the `test video` folder:
- `cctv1.mp4` — Normal activity
- `cctv2.mp4` — Fight incident

Upload either video via the **"Pilih Fail Video"** (Choose Video File) button on the dashboard to see the system in action.

##  Future Improvements

- Live RTSP/WebRTC streaming support for real-time detection (currently offline video analysis only)
- Edge deployment on embedded hardware (NVIDIA Jetson Nano, Raspberry Pi)
- Expand detection scope to other crime categories (theft, vandalism)

##  Acknowledgments

Special thanks to my supervisor, **Assoc. Prof. Ts. Dr. Nor Samsiah Sani**, for her guidance throughout this project.

##  License

This project was developed for academic purposes as part of a Final Year Project at FTSM, UKM.
