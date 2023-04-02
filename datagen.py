# Import necessary libraries
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
import os
import glob
# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
# Define the path to the directory containing the audio files
audio_dir = 'audiodata'

# Loop through each audio file and label it as crying or not crying
audio_files = []
labels = []
for file in glob.glob(os.path.join(audio_dir, '*.wav')):
    # If the file name contains the word 'cry', label it as crying
    if 'cry' in file:
        labels.append(1)
    # Otherwise, label it as not crying
    else:
        labels.append(0)
    # Append the file path to the list of audio files
    audio_files.append(file)
# Define the function to load and preprocess the audio data

def preprocess_audio_data(audio_files, labels, sample_rate=22050, n_mfcc=13, n_fft=2048, hop_length=512):
    # Initialize the arrays to store the audio data and labels
    X = []
    y = []
    # Loop through each audio file and load and preprocess the audio data
    for audio_file, label in zip(audio_files, labels):
        # Load the audio data using librosa
        audio, sr = librosa.load(audio_file, sr=sample_rate)

        # Extract the MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

        # Resize the MFCC features to a fixed size of (64, 64)
        mfccs = np.resize(mfccs, (64, 64))

        # Append the MFCC features and label to the arrays
        X.append(mfccs)
        y.append(label)

    # Convert the arrays to numpy arrays
    X = np.array(X)
    y = np.array(y)
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Add an extra dimension to the arrays for the CNN model
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    return X_train, X_test, y_train, y_test

x_train,x_test,y_train,y_test = preprocess_audio_data(audio_files,labels)


###   ---------------MODEL GENERATION------------------

# Define the model architecture
model = Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Load the audio data
# Here you can use a library like librosa to load and preprocess the audio data

# Train the model
model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=10,
    validation_data=(x_test, y_test)
)

# Save the model
model.save('cry_detection_model.h5')