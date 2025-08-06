#librosa is a python package for music and audio analysis necessary to create music information retrieval systems

import librosa
import librosa.display
import matplotlib.pyplot as plt

#waveform
y, sr = librosa.load('music1.wav')
plt.figure(figsize=(10, 4))
librosa.display.waveshow(y, sr=sr)
plt.title('Waveform Representation')
plt.show()

# Spectrogram
X = librosa.stft(y)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(10, 4))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()
plt.title('Spectrogram Representation')
plt.show()
