# Live ASL Virtual Webcam Transcription

A real-time American Sign Language (ASL) transcription pipeline using PyTorch and MediaPipe to broadcast live, dynamically centered captions directly into video calls via a virtual webcam.

## Features

* **Real-Time Translation:** Translates continuous ASL gestures into text using a custom PyTorch LSTM model.
* **Virtual Webcam Broadcast:** Pipes live video with burned-in captions directly into Google Meet, Zoom, and Teams using `pyvirtualcam`.
* **Dynamic Centered Captions:** Automatically calculates text width to keep live transcriptions perfectly centered at the bottom of the broadcast feed.
* **Robust Tracking Architecture:** Utilizes wrist-relative geometry, a 5.0x finger feature multiplier, anatomical sanity filters, and posture gating to eliminate transitional tracking hallucinations.
* **Anti-Spam Smoothing:** Employs a strict 17/20 frame majority-vote window with memory flushing to prevent double-typing.

## Prerequisites

### 1. Python Environment
This project requires **Python 3.11** (due to MediaPipe compatibility requirements). 

### 2. Virtual Camera Driver (macOS)
To broadcast to video conferencing apps, your Mac needs a signed virtual camera driver. 
1. Download and install [OBS Studio](https://obsproject.com/).
2. Open OBS, grant camera permissions, and click **Start Virtual Camera**.
3. Click **Stop Virtual Camera** and completely **Quit OBS**. (The app does not need to remain open; the Python script will hijack the installed driver).

## Installation

1. Clone the repository:
```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME
```

2. Create and activate a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install the required dependencies:
```bash
pip install opencv-python numpy mediapipe==0.10.21 torch pyautogui pyvirtualcam
```

## Usage

The pipeline is managed through a single engine script with three operating modes.

### 1. Data Collection
Collect your custom ASL signs or the `idle` background class using your webcam. The script features "Smart Skip" and will bypass any words that already have data in the `ASL_Data_WLASL` directory.
```bash
python asl_engine.py --mode collect
```

### 2. Model Training
Train the PyTorch LSTM neural network on the extracted landmark sequences. The model will automatically save as `asl_model.pth`.
```bash
python asl_engine.py --mode train
```

### 3. Live Inference & Broadcasting
Start the live transcription engine and broadcast the feed to your virtual camera.
```bash
python asl_engine.py --mode run
```

## Connecting to Google Meet / Zoom

1. Run the inference script: `python asl_engine.py --mode run`.
2. Look at the local OpenCV popup window on your desktop to monitor your signing (it acts as a readable mirror).
3. Open your video call platform's settings.
4. Change your camera input to **OBS Virtual Camera**.
5. *Note:* If your self-preview in Google Meet/Zoom looks inverted, ignore it. Video conferencing platforms locally mirror your preview, but your audience will see the un-flipped, correctly oriented text.
