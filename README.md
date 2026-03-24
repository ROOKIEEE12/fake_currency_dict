# Fake Currency Detection System

This project is a web-based application for detecting fake currency using a Deep Learning CNN model (MobileNetV2).

## Technologies Used
- **Python 3.10**
- **TensorFlow / Keras** (Deep Learning)
- **Flask** (Web Backend)
- **HTML/CSS/JavaScript** (Frontend)

## Project Structure
- `dataset/`: Folder for training images (Real/Fake).
- `static/`: CSS and JavaScript files.
- `templates/`: HTML templates.
- `app.py`: Flask application server.
- `train_currency_model.py`: Script to train the model.
- `requirements.txt`: List of dependencies.
- `currency_model.h5`: The trained model file (generated after training).

## Setup Instructions

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Train the Model (If not already trained):**
    Ensure you have `dataset/Real` and `dataset/Fake` folders with images.
    ```bash
    python train_currency_model.py
    ```

3.  **Run the Web App:**
    ```bash
    python app.py
    ```

4.  **Access the App:**
    Open your browser and go to: `http://127.0.0.1:5000`

## Features
- **Image Upload:** Upload a photo of a currency note to check it.
- **Webcam Support:** Use your webcam to scan a note in real-time.
- **Instant Analysis:** Get immediate "Real" or "Fake" results with confidence scores.
