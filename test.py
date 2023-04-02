import librosa
import numpy as np
import os

# Set up paths
data_dir = 'cryingbaby'  # Directory containing audio files
save_dir = 'pdata'  # Directory to save preprocessed data
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Set up parameters
n_fft = 2048  # Number of FFT points
hop_length = 512  # Hop length for FFT
n_mels = 128  # Number of Mel bands
sample_rate = 22050  # Sampling rate of audio files
duration = 4  # Desired duration of preprocessed audio segments in seconds
samples_per_segment = sample_rate * duration

# Load audio files and generate preprocessed data
for i, filename in enumerate(os.listdir(data_dir)):
    if filename.endswith('.wav'):
        print(f'Processing file {i+1}: {filename}')
        # Load audio file
        file_path = os.path.join(data_dir, filename)
        y, sr = librosa.load(file_path, sr=sample_rate)
        # Generate preprocessed data
        for j in range(0, len(y) - samples_per_segment + 1, samples_per_segment):
            # Extract segment
            segment = y[j:j + samples_per_segment]
            # Compute mel spectrogram
            spec = librosa.feature.melspectrogram(y=segment, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
            spec_db = librosa.power_to_db(spec, ref=np.max)
            # Normalize between -1 and 1
            spec_norm = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min()) * 2 - 1
            # Save preprocessed data
            save_path = os.path.join(save_dir, f'{filename[:-4]}_{j}.npy')
            np.save(save_path, spec_norm)
