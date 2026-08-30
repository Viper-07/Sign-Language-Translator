import numpy as np

def extract_landmarks(hand):
    """
    Extracts and normalizes landmarks from a detected hand dictionary.
    Returns a flattened list of (x, y) coordinates.
    """
    lm_list = hand["lmList"]
    
    if not lm_list:
        return None
        
    # Extract only x,y coords
    coords = np.array([[lm[1], lm[2]] for lm in lm_list])
    
    # Normalize by shifting the minimum x and y to 0
    min_x = np.min(coords[:, 0])
    min_y = np.min(coords[:, 1])
    
    normalized_coords = coords - [min_x, min_y]
    
    # Flatten the list
    flattened_coords = normalized_coords.flatten().tolist()
    
    return flattened_coords
