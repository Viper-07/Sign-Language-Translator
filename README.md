# 🤟 AI Sign Language Translator

A real-time sign language recognition system powered by **MediaPipe**, **OpenCV**, and a **Neural Network (MLPClassifier)**. Point your webcam at your hand and the app instantly predicts the sign you're making.

---

## ✨ Features

- 📷 **Real-time hand detection** using MediaPipe's landmark model
- 🧠 **Neural Network classifier** (Deep MLP) trained on hand landmark data
- 🔡 **Custom sign support** — teach the model your own gestures in minutes
- 📊 **Sentence engine** with temporal smoothing for stable, flicker-free predictions
- 🚀 **Pre-trained data download** — get started instantly without collecting data manually

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Hand Detection | [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands) |
| Computer Vision | [OpenCV](https://opencv.org/) |
| AI / ML Model | [Scikit-learn MLPClassifier](https://scikit-learn.org/) |
| Language | Python 3.11 |

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/Sign_Language_Translator.git
cd Sign_Language_Translator
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start (Pre-trained Data)

Skip data collection entirely — download an open-source ASL landmark dataset and train instantly:

```bash
python src/download_pretrained_data.py
python src/train_model.py
python app.py
```

---

## 🎓 Train on Your Own Signs (Recommended for Best Accuracy)

### Step 1 — Collect data
Run the collection script and follow the prompts:
```bash
python src/collect_data.py
```
- Enter a label (e.g. `A`, `B`, `Hello`, `Fist`)
- Press **`S`** to start/stop recording your sign
- Press **`Q`** to quit and save
- Repeat for each sign you want to teach the model

### Step 2 — Train the model
```bash
python src/train_model.py
```

### Step 3 — Run the app
```bash
python app.py
```
Press **`Q`** in the OpenCV window to exit.

---

## 📁 Project Structure

```
Sign_Language_Translator/
│
├── app.py                        # Main application entry point
├── requirements.txt              # Python dependencies
│
├── src/
│   ├── hand_detector.py          # MediaPipe hand landmark detection
│   ├── landmark_extractor.py     # Normalizes landmarks for model input
│   ├── predictor.py              # Loads model and runs inference
│   ├── sentence_engine.py        # Temporal smoothing for stable predictions
│   ├── collect_data.py           # Webcam-based data collection tool
│   ├── train_model.py            # Trains the MLP Neural Network
│   └── download_pretrained_data.py  # Downloads open-source ASL dataset
│
├── data/
│   └── raw/                      # Collected CSV landmark data
│
└── models/
    └── sign_model.pkl            # Trained model (generated after training)
```

---

## 🤝 Contributing

Pull requests are welcome! If you'd like to add support for more signs, improve the model architecture, or enhance the UI, feel free to open an issue or PR.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
