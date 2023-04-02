import os
import glob

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