from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load model once
model = tf.keras.models.load_model('face_emotionModel.h5')

# Labels (match your model)
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def preprocess_image(file_path):
    """Read, resize, normalize, and reshape image for model prediction."""
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Change to IMREAD_COLOR if needed
    img = cv2.resize(img, (48, 48))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)      # batch dimension
    img = np.expand_dims(img, axis=-1)     # channel dimension (for grayscale)
    return img

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
    temp_path = os.path.join('static', 'temp.jpg')
    file.save(temp_path)

    try:
        # Preprocess & predict
        img = preprocess_image(temp_path)
        prediction = model.predict(img)
        emotion = emotions[np.argmax(prediction)]
    except Exception as e:
        os.remove(temp_path)
        return jsonify({'error': str(e)})
    
    # Remove temporary file
    os.remove(temp_path)

    return jsonify({'emotion': emotion})

if __name__ == '__main__':
    app.run(debug=True)
