import cv2
import os
import pandas as pd
import sys
# Add parent directory to path to allow importing from src when run as a script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hand_detector import HandDetector
from src.landmark_extractor import extract_landmarks

def collect_data():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(max_hands=1)
    
    data = []
    labels = []
    
    label = input("Enter the label for the sign you are about to perform: ")
    print("Press 's' to start/stop recording. Press 'q' to quit.")
    
    recording = False
    
    while True:
        success, img = cap.read()
        if not success:
            break
            
        img = cv2.flip(img, 1)
        hands, img = detector.find_hands(img)
        
        if hands and recording:
            hand = hands[0]
            landmarks = extract_landmarks(hand)
            if landmarks:
                data.append(landmarks)
                labels.append(label)
                cv2.putText(img, "Recording...", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("Data Collection", img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            recording = not recording
            if recording:
                print("Started recording.")
            else:
                print("Stopped recording.")
                
    cap.release()
    cv2.destroyAllWindows()
    
    if data:
        df = pd.DataFrame(data)
        df['label'] = labels
        
        # Ensure we save relative to the project root, not the script location
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        raw_dir = os.path.join(root_dir, 'data', 'raw')
        os.makedirs(raw_dir, exist_ok=True)
        file_path = os.path.join(raw_dir, f'{label}_data.csv')
        
        # Append if exists
        if os.path.exists(file_path):
            df.to_csv(file_path, mode='a', header=False, index=False)
        else:
            df.to_csv(file_path, index=False)
            
        print(f"Saved {len(data)} samples to {file_path}")

if __name__ == "__main__":
    collect_data()
