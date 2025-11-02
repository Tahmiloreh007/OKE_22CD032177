# model_training.py

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# ---------------------------------------------
# STEP 1: Load and preprocess the dataset
# ---------------------------------------------
print("📂 Loading FER2013 dataset...")
data = pd.read_csv('fer2013.csv')

# Each image is a string of pixel values; split them into arrays
pixels = data['pixels'].tolist()
X = np.array([np.fromstring(p, sep=' ') for p in pixels], dtype='float32')

# Normalize pixel values to 0–1 range
X = X / 255.0

# Reshape for CNN input (48x48 grayscale images)
X = X.reshape(-1, 48, 48, 1)

# Convert emotion labels to categorical (one-hot encoding)
y = to_categorical(data['emotion'], num_classes=7)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("✅ Data loaded and preprocessed successfully!")
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# ---------------------------------------------
# STEP 2: Build the CNN model
# ---------------------------------------------
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 emotions
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ---------------------------------------------
# STEP 3: Train the model
# ---------------------------------------------
print("🚀 Training the model (this may take several minutes)...")
history = model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test))

# ---------------------------------------------
# STEP 4: Save the trained model
# ---------------------------------------------
model.save('face_emotionModel.h5')
print("✅ Model training complete and saved as face_emotionModel.h5")
