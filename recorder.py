
# import required libraries
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
import time
import os
 
# Sampling frequency
freq = 44100
 
# Recording duration
duration = 5
#recording directory
directory = "recordings/"
n=0
while True:
    if n==9:
        n=0
    # Start recorder with the given values
    # of duration and sample frequency
    recording = sd.rec(int(duration * freq),
                    samplerate=freq, channels=2)
    
    # Record audio for the given number of seconds
    sd.wait()
    
    # This will convert the NumPy array to an audio
    # file with the given sampling frequency
    # write(f"{directory}recording0.wav", freq, recording)
    
    # Convert the NumPy array to audio file
    wv.write(f"{directory}recording{str(n)}.wav", recording, freq, sampwidth=2)
    n+=1
    time.sleep(1)
    # os.remove(f"{directory}recording0.wav")