import librosa
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('cry_detection_model.h5')

# Define the function to predict the cry sound
def predict_cry_sound(audio_file_path):
    # Load and preprocess the audio data
    audio, sr = librosa.load(audio_file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
    mfccs = np.resize(mfccs, (64, 64, 1))
    mfccs = np.expand_dims(mfccs, axis=0)

    # Predict the cry sound
    prediction = model.predict(mfccs)

    # Return the predicted label
    if prediction[0][0] > 0.5:
        return 'Crying'
    else:
        return 'Not crying'

# Test the model on an audio file
audio_file_path = 'recording1.wav'
prediction = predict_cry_sound(audio_file_path)
print('The audio file is:', prediction)
