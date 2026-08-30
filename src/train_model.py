import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def train():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_dir = os.path.join(root_dir, 'data', 'raw')
    
    if not os.path.exists(raw_data_dir) or not os.listdir(raw_data_dir):
        print("No raw data found. Please run collect_data.py first.")
        return
        
    all_data = []
    
    # Load all csv files in the raw directory
    for file in os.listdir(raw_data_dir):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(raw_data_dir, file))
            all_data.append(df)
            
    if not all_data:
        print("No CSV files found in the raw data directory.")
        return
        
    combined_df = pd.concat(all_data, ignore_index=True)
    
    X = combined_df.iloc[:, :-1].values # Features
    y = combined_df.iloc[:, -1].values  # Labels
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model trained with accuracy: {accuracy * 100:.2f}%")
    
    models_dir = os.path.join(root_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, 'sign_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train()
