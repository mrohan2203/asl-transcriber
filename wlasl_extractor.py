import os
import json
import cv2
import numpy as np
import mediapipe as mp

# --- CONFIGURATION ---
WLASL_JSON_PATH = '/Users/rohanmuru/Downloads/archive (6)/WLASL_v0.3.json'
VIDEO_DIR = '/Users/rohanmuru/Downloads/archive (6)/videos'
OUTPUT_DATA_PATH = 'ASL_Data_WLASL'

# Target vocabulary
TARGET_WORDS = ['hello', 'thanks', 'iloveyou', 'drink', 'computer', 'book', 'apple'] 
SEQ_LENGTH = 20

mp_holistic = mp.solutions.holistic

# --- GLOBAL CACHE FOR FORWARD FILLING ---
last_known_lh = np.zeros(21*3)
last_known_rh = np.zeros(21*3)

def extract_robust_features(results):
    """Universal feature extractor with wrist-relative geometry and forward-filling."""
    global last_known_lh, last_known_rh

    # 1. Macro Movement: Pose relative to Nose
    if results.pose_landmarks:
        nose = results.pose_landmarks.landmark[0]
        pose = np.array([[res.x - nose.x, res.y - nose.y, res.z - nose.z, res.visibility] 
                         for res in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33*4)

    # 2. Micro Movement: Left Hand relative to Left Wrist
    if results.left_hand_landmarks:
        wrist = results.left_hand_landmarks.landmark[0]
        lh = np.array([[res.x - wrist.x, res.y - wrist.y, res.z - wrist.z] 
                       for res in results.left_hand_landmarks.landmark]).flatten()
        last_known_lh = lh
    else:
        lh = last_known_lh

    # 3. Micro Movement: Right Hand relative to Right Wrist
    if results.right_hand_landmarks:
        wrist = results.right_hand_landmarks.landmark[0]
        rh = np.array([[res.x - wrist.x, res.y - wrist.y, res.z - wrist.z] 
                       for res in results.right_hand_landmarks.landmark]).flatten()
        last_known_rh = rh
    else:
        rh = last_known_rh

    return np.concatenate([pose, lh, rh])

def process_video(video_path, save_dir):
    global last_known_lh, last_known_rh
    
    # Reset cache for every new video to prevent bleeding between files
    last_known_lh = np.zeros(21*3)
    last_known_rh = np.zeros(21*3)

    cap = cv2.VideoCapture(video_path)
    frames_data = []
    
    with mp_holistic.Holistic(min_detection_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            frames_data.append(extract_robust_features(results))
            
    cap.release()
    
    if len(frames_data) == 0:
        return False
        
    # Standardize to exactly SEQ_LENGTH
    if len(frames_data) > SEQ_LENGTH:
        indices = np.linspace(0, len(frames_data) - 1, SEQ_LENGTH, dtype=int)
        frames_data = [frames_data[i] for i in indices]
    else:
        padding = SEQ_LENGTH - len(frames_data)
        frames_data.extend([frames_data[-1]] * padding)
        
    os.makedirs(save_dir, exist_ok=True)
    for i, frame_data in enumerate(frames_data):
        np.save(os.path.join(save_dir, f"{i}.npy"), frame_data)
        
    return True

def build_dataset():
    print("Loading WLASL JSON...")
    with open(WLASL_JSON_PATH, 'r') as f:
        wlasl_data = json.load(f)
        
    for entry in wlasl_data:
        word = entry['gloss']
        if word not in TARGET_WORDS:
            continue
            
        print(f"\nProcessing word: {word}")
        successful_videos = 0
        
        for instance in entry['instances']:
            video_id = instance['video_id']
            video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
            
            if not os.path.exists(video_path):
                continue
                
            save_dir = os.path.join(OUTPUT_DATA_PATH, word, str(successful_videos))
            success = process_video(video_path, save_dir)
            if success:
                successful_videos += 1
                print(f"  [+] Extracted {video_id}.mp4")

    print("\nDataset extraction complete.")

if __name__ == '__main__':
    build_dataset()