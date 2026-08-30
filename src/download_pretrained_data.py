import urllib.request
import pandas as pd
import os

def download_dataset():
    print("Downloading pre-trained landmark dataset...")
    url = 'https://raw.githubusercontent.com/kinivi/hand-gesture-recognition-mediapipe/main/model/keypoint_classifier/keypoint.csv'
    
    try:
        response = urllib.request.urlopen(url)
        data = response.read().decode('utf-8')
    except Exception as e:
        print(f"Failed to download dataset: {e}")
        return

    # Parse CSV data
    lines = [line.strip().split(',') for line in data.split('\n') if line.strip()]
    
    # Kinivi's dataset has label as the first column. Our model expects label as the last column.
    # It contains 42 features (x, y for 21 landmarks)
    processed_data = []
    
    # Map numeric gestures to ASL alphabet letters for demonstration
    # 0: Open, 1: Close, 2: Pointer, 3: OK
    label_map = {
        '0': 'A', # Mapping Open hand to A
        '1': 'B', # Mapping Close hand to B
        '2': 'C', # Mapping Pointer to C
        '3': 'D'  # Mapping OK to D
    }
    
    for row in lines:
        if len(row) == 43:
            label = row[0]
            features = row[1:]
            
            # Use mapped label, default to 'E' for others
            mapped_label = label_map.get(label, 'E')
            
            # Reconstruct row: features + label
            new_row = features + [mapped_label]
            processed_data.append(new_row)

    if processed_data:
        df = pd.DataFrame(processed_data)
        
        # Save to data/raw/
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        raw_dir = os.path.join(root_dir, 'data', 'raw')
        os.makedirs(raw_dir, exist_ok=True)
        
        file_path = os.path.join(raw_dir, 'pretrained_asl_data.csv')
        df.to_csv(file_path, index=False, header=False)
        print(f"Successfully downloaded and processed {len(processed_data)} samples to {file_path}")
    else:
        print("No valid data found.")

if __name__ == "__main__":
    download_dataset()
