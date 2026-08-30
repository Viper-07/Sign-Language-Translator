import pickle
import os
import numpy as np

class Predictor:
    def __init__(self, model_path="models/sign_model.pkl"):
        # Ensure path is relative to project root if not absolute
        if not os.path.isabs(model_path):
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(root_dir, model_path)
            
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            self.model_loaded = True
        except FileNotFoundError:
            print(f"Warning: Model not found at {model_path}")
            print("Please train the model first.")
            self.model_loaded = False
            
    def predict(self, landmarks):
        if not self.model_loaded or not landmarks:
            return ""
            
        # Landmarks should be flattened 1D list, model expects 2D array
        landmarks_array = np.array(landmarks).reshape(1, -1)
        
        prediction = self.model.predict(landmarks_array)
        return prediction[0]
