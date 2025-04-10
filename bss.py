import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import os


def separate(fpath):
    with wave.open(fpath, "rb") as w:
    # Extract parameters of .wav file
        n_channels = w.getnchannels()
        samp_width = w.getsampwidth()   
        framerate = w.getframerate()
        n_frames = w.getnframes()
    
        raw_data = w.readframes(n_frames)

    # Convert raw data to numpy array
    if samp_width == 1:
        dtype = np.int8
    elif samp_width == 2:
        dtype = np.int16
    elif samp_width == 3:
        dtype = np.int32
    elif samp_width == 4:
        dtype = np.int32
    else:
        raise ValueError("Unsupported sample width: {}".format(samp_width))
    
    audio_data = np.frombuffer(raw_data, dtype=dtype)
    audio_data = audio_data.astype(np.float32)

    # Taking STFT

    window_size = 1024
    overlap = 512

    frequencies, time_segments, stft_matrix = stft(
        audio_data,
        fs=framerate,
        window='hann',
        nperseg=window_size,
        noverlap=overlap,
        nfft=window_size,
        return_onesided=True
    )

    
    print(frequencies, time_segments, stft_matrix)

    magnitude_spectrogram = np.abs(stft_matrix)

    # Create a mask that selects only time up to 10 seconds
    time_mask = time_segments <= 10.0

    # Subset the time axis and the spectrogram data
    time_segments_10 = time_segments[time_mask]
    magnitude_spectrogram_10 = magnitude_spectrogram[:, time_mask]

    # Plot only the first 10 seconds
    plt.figure()
    plt.pcolormesh(time_segments_10, frequencies, magnitude_spectrogram_10, shading='gouraud')
    plt.title("Spectrogram - First 10 Seconds")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Magnitude")
    plt.show()

'''
    plt.figure()
    plt.pcolormesh(time_segments, frequencies, magnitude_spectrogram, shading='gouraud')
    plt.title("Input File Spectrogram")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Magnitude")
    plt.show()
'''


