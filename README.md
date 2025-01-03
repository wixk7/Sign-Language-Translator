# Sign Language Translator

This project is a real-time sign language translator that uses computer vision and machine learning to recognize hand gestures and translate them into letters of the alphabet.

## Features
- **Data Collection**: Capture hand gesture images for training.
- **Dataset Creation**: Preprocess the collected data into a format suitable for model training.
- **Model Training**: Train a machine learning model to recognize sign language gestures.
- **Live Translation**: Use a webcam to translate hand gestures into corresponding letters in real time.

---

## Requirements
Ensure the following dependencies are installed:

- Python 3.8+
- OpenCV
- MediaPipe
- NumPy
- Scikit-learn

Install dependencies using:
```bash
pip install opencv-python mediapipe numpy scikit-learn
```

---

## How to Use

### 1. **Data Collection**
Run the `imgcollection.py` script to collect hand gesture images for training.
```bash
python imgcollection.py
```
- Follow the prompts to capture images for each class.
- Images will be saved in the `./data` directory.

### 2. **Dataset Creation**
Process the collected images into a format suitable for training using `datasetcreation.py`.
```bash
python datasetcreation.py
```
- Features and labels will be saved into `data.pickle`.

### 3. **Model Training**
Train the model using `training.py`.
```bash
python training.py
```
- The trained model will be saved as `model.p`.

### 4. **Live Translation**
Run the `inferenceclassfier.py` script to start translating sign language gestures in real time.
```bash
python inferenceclassfier.py
```

---

## File Descriptions

### `imgcollection.py`
- Collects images of hand gestures for each class.
- Saves images in the `./data` directory.

### `datasetcreation.py`
- Processes collected images and extracts features using MediaPipe.
- Saves features and labels into `data.pickle`.

### `training.py`
- Loads `data.pickle` and trains a Random Forest Classifier.
- Saves the trained model as `model.p`.

### `inferenceclassfier.py`
- Uses the trained model to translate hand gestures captured via webcam into letters.

---

## Project Structure
```plaintext
.
├── imgcollection.py      # Script for collecting gesture images
├── datasetcreation.py    # Script for creating the dataset
├── training.py           # Script for training the model
├── inferenceclassfier.py # Script for live translation
├── data/                 # Directory to store images and processed data
└── model.p               # Trained model file (generated after training)
```

---

## Contribution
Contributions are welcome! Feel free to fork the repository and submit pull requests.

---

## Acknowledgments
- **MediaPipe**: Used for hand tracking.
- **Scikit-learn**: Used for training the Random Forest model.
- Inspiration from various open-source sign language recognition projects.
