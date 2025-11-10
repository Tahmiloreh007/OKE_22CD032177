from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import cv2
import os
import sqlite3
from datetime import datetime

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model('face_emotionModel.h5')

# Emotion labels (adjust if different)
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ----------------------------
# Database connection function
# ----------------------------
def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

# ----------------------------
# Homepage route
# ----------------------------
@app.route('/')
def home():
    return render_template('index.html')

# ----------------------------
# Emotion prediction route
# ----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty file name'})

    # Save image temporarily
    path = os.path.join('static', 'temp.jpg')
    file.save(path)

    try:
        # Read and preprocess image
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (48, 48))
        img = img / 255.0
        img = np.reshape(img, (1, 48, 48, 1))

        # Predict emotion
        prediction = model.predict(img)
        emotion = emotions[np.argmax(prediction)]

        # Save prediction to database
        conn = get_db_connection()
        conn.execute(
            'INSERT INTO predictions (username, emotion, image_path, timestamp) VALUES (?, ?, ?, ?)',
            ('Anonymous User', emotion, path, datetime.now())
        )
        conn.commit()
        conn.close()

        # Remove temporary file
        os.remove(path)

        return jsonify({'emotion': emotion})

    except Exception as e:
        if os.path.exists(path):
            os.remove(path)
        return jsonify({'error': str(e)})

# ----------------------------
# Run the app
# ----------------------------
if __name__ == '__main__':
    app.run(debug=True)
