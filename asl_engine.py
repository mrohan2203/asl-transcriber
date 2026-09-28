import os
import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import argparse
import pyautogui
import pyvirtualcam

# --- CONFIGURATION ---
ACTIONS = np.array(['hello', 'thanks', 'iloveyou', 'idle', 'drink', 'computer', 'book', 'apple'])
SEQ_LENGTH = 20      
NUM_SEQUENCES = 30   
DATA_PATH = os.path.join('ASL_Data_WLASL') # Pointing to hybrid data folder

# --- PYTORCH MODEL ---
class ASLSequenceModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size=258, hidden_size=128, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc1(out[:, -1, :]) 
        out = self.relu(out)
        out = self.fc2(out)
        return out

# --- MEDIAPIPE PIPELINE & CACHE ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Global Cache & Timers for Forward Filling
last_known_lh = np.zeros(21*3)
last_known_rh = np.zeros(21*3)
lh_missing_count = 0
rh_missing_count = 0

def extract_robust_features(results):
    global last_known_lh, last_known_rh, lh_missing_count, rh_missing_count

    # 1. Pose relative to Nose
    if results.pose_landmarks:
        nose = results.pose_landmarks.landmark[0]
        pose = np.array([[res.x - nose.x, res.y - nose.y, res.z - nose.z, res.visibility] 
                         for res in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33*4)

    # --- ANATOMICAL SANITY CHECK FUNCTION ---
    def is_hand_sane(hand_landmarks):
        wrist = hand_landmarks.landmark[0]
        for lm in hand_landmarks.landmark:
            dist = np.sqrt((lm.x - wrist.x)**2 + (lm.y - wrist.y)**2)
            # The Goldilocks Zone: 0.55
            if dist > 0.55: 
                return False
        return True
    # ----------------------------------------

    # 2. Left Hand
    if results.left_hand_landmarks and is_hand_sane(results.left_hand_landmarks):
        wrist = results.left_hand_landmarks.landmark[0]
        lh = np.array([[res.x - wrist.x, res.y - wrist.y, res.z - wrist.z] 
                       for res in results.left_hand_landmarks.landmark]).flatten()
        lh = lh * 5.0  
        last_known_lh = lh
        lh_missing_count = 0
    else:
        lh_missing_count += 1
        # EXTENDED MEMORY: Remember the hand shape for a full 20 frames (0.6 seconds)
        if lh_missing_count > 20:
            last_known_lh = np.zeros(21*3)
        lh = last_known_lh

    # 3. Right Hand
    if results.right_hand_landmarks and is_hand_sane(results.right_hand_landmarks):
        wrist = results.right_hand_landmarks.landmark[0]
        rh = np.array([[res.x - wrist.x, res.y - wrist.y, res.z - wrist.z] 
                       for res in results.right_hand_landmarks.landmark]).flatten()
        rh = rh * 5.0  
        last_known_rh = rh
        rh_missing_count = 0
    else:
        rh_missing_count += 1
        # EXTENDED MEMORY: Remember the hand shape for a full 20 frames (0.6 seconds)
        if rh_missing_count > 20:
            last_known_rh = np.zeros(21*3)
        rh = last_known_rh

    return np.concatenate([pose, lh, rh])

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

# --- MODE 1: DATA COLLECTION ---
def collect_data():
    global last_known_lh, last_known_rh, lh_missing_count, rh_missing_count

    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in ACTIONS:
            action_path = os.path.join(DATA_PATH, action)
            os.makedirs(action_path, exist_ok=True)
            
            # SMART SKIP: If any data exists, skip recording
            if len(os.listdir(action_path)) > 0:
                print(f"Data already exists for '{action}'. Skipping.")
                continue
                
            print(f"\n--- Get ready to sign: {action.upper()} ---")
            cv2.waitKey(2000)
            
            for sequence in range(NUM_SEQUENCES):
                last_known_lh = np.zeros(21*3)
                last_known_rh = np.zeros(21*3)
                lh_missing_count = 0
                rh_missing_count = 0
                
                seq_path = os.path.join(action_path, str(sequence))
                os.makedirs(seq_path, exist_ok=True)
                
                for frame_num in range(SEQ_LENGTH):
                    ret, frame = cap.read()
                    if not ret: break
                        
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic.process(image)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    draw_styled_landmarks(image, results)
                    
                    if frame_num == 0: 
                        cv2.putText(image, f'STARTING COLLECTION: {action}', (120,200), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                        cv2.imshow('ASL Data Collector', image)
                        cv2.waitKey(500) 
                    else: 
                        cv2.putText(image, f'Recording seq {sequence}/{NUM_SEQUENCES}', (15,12), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('ASL Data Collector', image)
                    
                    keypoints = extract_robust_features(results)
                    npy_path = os.path.join(seq_path, str(frame_num))
                    np.save(npy_path, keypoints)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
    cap.release()
    cv2.destroyAllWindows()

# --- MODE 2: TRAINING ---
def train_model():
    sequences, labels = [], []
    action_map = {label:num for num, label in enumerate(ACTIONS)}
    
    print("Loading robust data from disk...")
    for action in ACTIONS:
        action_path = os.path.join(DATA_PATH, action)
        if not os.path.exists(action_path):
            print(f"Warning: Missing data for {action}")
            continue
            
        for sequence_dir in os.listdir(action_path):
            seq_path = os.path.join(action_path, sequence_dir)
            if not os.path.isdir(seq_path): continue
                
            window = []
            for frame_num in range(SEQ_LENGTH):
                frame_path = os.path.join(seq_path, f"{frame_num}.npy")
                if os.path.exists(frame_path):
                    window.append(np.load(frame_path))
                    
            if len(window) == SEQ_LENGTH:
                sequences.append(window)
                labels.append(action_map[action])
            
    X = torch.tensor(np.array(sequences), dtype=torch.float32)
    y = torch.tensor(np.array(labels), dtype=torch.long)
    
    model = ASLSequenceModel(num_classes=len(ACTIONS))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Training PyTorch LSTM on {len(X)} sequences...")
    for epoch in range(150):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/150, Loss: {loss.item():.4f}')
            
    torch.save(model.state_dict(), 'asl_model.pth')
    print("Model saved to asl_model.pth")

# --- MODE 3: REAL-TIME INFERENCE ---
def run_inference():
    global last_known_lh, last_known_rh, lh_missing_count, rh_missing_count
    last_known_lh = np.zeros(21*3)
    last_known_rh = np.zeros(21*3)
    lh_missing_count = 0
    rh_missing_count = 0

    model = ASLSequenceModel(num_classes=len(ACTIONS))
    model.load_state_dict(torch.load('asl_model.pth'))
    model.eval()
    
    sequence_buffer = deque(maxlen=SEQ_LENGTH)
    sentence = []
    prediction_history = [] 
    
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic, \
         pyvirtualcam.Camera(width=width, height=height, fps=30) as cam:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
                
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            draw_styled_landmarks(image, results)
            
            keypoints = extract_robust_features(results)
            sequence_buffer.append(keypoints)
            
            if len(sequence_buffer) == SEQ_LENGTH:
                input_tensor = torch.tensor(np.array(sequence_buffer), dtype=torch.float32).unsqueeze(0)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.softmax(output, dim=1).squeeze()
                    confidence, predicted_idx = torch.max(probabilities, dim=0)
                    predicted_action = ACTIONS[predicted_idx.item()]
                
                # --- SMART POSTURE GATE (UPGRADED) ---
                is_hands_in_frame = False
                if results.pose_landmarks:
                    left_shoulder = results.pose_landmarks.landmark[11].y
                    right_shoulder = results.pose_landmarks.landmark[12].y
                    avg_shoulder = (left_shoulder + right_shoulder) / 2.0
                    
                    l_wrist = results.pose_landmarks.landmark[15]
                    r_wrist = results.pose_landmarks.landmark[16]
                    
                    if (l_wrist.visibility > 0.5 and l_wrist.y < avg_shoulder + 0.25) or \
                       (r_wrist.visibility > 0.5 and r_wrist.y < avg_shoulder + 0.25):
                        is_hands_in_frame = True

                if not is_hands_in_frame:
                    predicted_action = 'idle'
                    confidence = torch.tensor(1.0)
                # --------------------------------------

                print(f"Raw Pred: {predicted_action} | Confidence: {confidence.item():.2f}")

                # --- ROBUST MAJORITY VOTE SMOOTHING ---
                prediction_history.append(predicted_action)
                if len(prediction_history) > 20: 
                    prediction_history = prediction_history[-20:]
                
                if len(prediction_history) > 0:
                    most_common_pred = max(set(prediction_history), key=prediction_history.count)
                else:
                    most_common_pred = 'idle'
                
                if prediction_history.count(most_common_pred) >= 17 and confidence.item() > 0.80:
                    if most_common_pred != 'idle':
                        if len(sentence) == 0 or most_common_pred != sentence[-1]: 
                            sentence.append(most_common_pred)
                            prediction_history.clear()
                            
                if len(sentence) > 5:
                    sentence = sentence[-5:]
                    
            # --- DYNAMIC BOTTOM-CENTERED CAPTIONS ---
            text = ' '.join(sentence)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2
            
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_width = text_size[0]
            
            text_x = int((width - text_width) / 2)
            text_y = int(height - 20)
            
            cv2.rectangle(image, (0, int(height - 60)), (int(width), int(height)), (245, 117, 16), -1)
            cv2.putText(image, text, (text_x, text_y), 
                       font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            # ----------------------------------------
            
            # --- LOCAL DESKTOP PREVIEW ---
            cv2.imshow('Live ASL Transcription', image)
            
            # --- VIRTUAL WEBCAM BROADCAST ---
            out_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cam.send(out_frame)
            cam.sleep_until_next_frame()
            # -----------------------------------
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Real-Time ASL Pipeline")
    parser.add_argument('--mode', type=str, required=True, choices=['collect', 'train', 'run'])
    args = parser.parse_args()
    
    if args.mode == 'collect': collect_data()
    elif args.mode == 'train': train_model()
    elif args.mode == 'run': run_inference()