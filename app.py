from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)
model = tf.keras.models.load_model('face_emotionModel.h5')

# Labels (adjust if your model uses different ones)
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty file name'})
    
    # Save temporarily
    path = os.path.join('static', 'temp.jpg')
    file.save(path)

    # Read and preprocess
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))
    img = img / 255.0
    img = np.reshape(img, (1, 48, 48, 1))

    # Predict
    prediction = model.predict(img)
    emotion = emotions[np.argmax(prediction)]

    # Remove temporary file
    os.remove(path)

    return jsonify({'emotion': emotion})

if __name__ == '__main__':
    app.run(debug=True)
