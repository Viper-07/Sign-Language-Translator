import collections

class SentenceEngine:
    def __init__(self, buffer_size=10, confidence_threshold=0.7):
        self.buffer = collections.deque(maxlen=buffer_size)
        self.confidence_threshold = confidence_threshold
        self.last_stable_word = ""
        
    def process(self, current_prediction):
        if not current_prediction:
            return self.last_stable_word
            
        self.buffer.append(current_prediction)
        
        # Check if the most common prediction in the buffer meets the threshold
        if len(self.buffer) == self.buffer.maxlen:
            counter = collections.Counter(self.buffer)
            most_common_pred, count = counter.most_common(1)[0]
            
            confidence = count / len(self.buffer)
            
            if confidence >= self.confidence_threshold:
                self.last_stable_word = most_common_pred
                
        return self.last_stable_word
