import cv2
from src.hand_detector import HandDetector
from src.landmark_extractor import extract_landmarks
from src.predictor import Predictor
from src.sentence_engine import SentenceEngine

def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    predictor = Predictor("models/sign_model.pkl")
    engine = SentenceEngine()

    while True:
        success, img = cap.read()
        if not success:
            break
            
        img = cv2.flip(img, 1)
        
        hands, img = detector.find_hands(img)
        
        if hands:
            # For simplicity, we process the first detected hand
            hand = hands[0]
            landmarks = extract_landmarks(hand)
            
            prediction = predictor.predict(landmarks)
            final_word = engine.process(prediction)
            
            # Display prediction
            cv2.putText(img, f"Prediction: {prediction}", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            cv2.putText(img, f"Word: {final_word}", (10, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
        cv2.imshow("AI Sign Language Translator", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
