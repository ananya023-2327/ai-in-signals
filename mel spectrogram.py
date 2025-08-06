import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Loading audio
y, sr = librosa.load('voice-assistants.wav')

# Compute Mel Spectrogram
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

# Convert to Decibel scale
S_dB = librosa.power_to_db(S, ref=np.max)

# Plot it
plt.figure(figsize=(12, 4))
librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
plt.colorbar(format="%+2.0f dB")
plt.title('Mel Spectrogram ( in dB)')
plt.tight_layout()
plt.show()
