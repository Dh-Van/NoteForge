import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import os
from utils import note_to_freq
import pickle


def separate(fpath, window_size=1024, overlap=None):
    # Load audio and force mono
    with wave.open(fpath, 'rb') as w:
        fs = w.getframerate()
        raw = w.readframes(w.getnframes())
        dtype = {1: np.int8, 2: np.int16, 4: np.int32}[w.getsampwidth()]
        audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        if audio.ndim > 1:
            audio = audio.reshape(-1, w.getnchannels())[:, 0]

    # Compute STFT
    if overlap is None:
        overlap = window_size // 2
    freqs, times, Zxx = stft(audio, fs=fs, nperseg=window_size, noverlap=overlap)
    S = np.abs(Zxx)

    # Define notes from A1 through G6 (flats only)
    notes = []
    # Octave 1: A1, Bb1, B1
    for tone in ["A", "Bb", "B"]:
        notes.append(f"{tone}1")
    # Octaves 2-5: full chromatic scale in flats
    for octave in range(2, 6):
        for tone in ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]:
            notes.append(f"{tone}{octave}")
    # Octave 6: C6 through G6
    for tone in ["C", "Db", "D", "Eb", "E", "F", "Gb", "G"]:
        notes.append(f"{tone}6")

    n_frames = S.shape[1]
    n_notes = len(notes)
    mag_matrix = np.zeros((n_frames, n_notes), dtype=np.float32)

    # Sum magnitudes around each note's frequency
    for j, nm in enumerate(notes):
        tone = nm[:-1]
        octave = nm[-1]
        f0 = note_to_freq(tone, octave)
        # half-semitone band
        f_low  = f0 * 2 ** (-1/24)
        f_high = f0 * 2 ** ( 1/24)
        idx = np.where((freqs >= f_low) & (freqs <= f_high))[0]
        if idx.size:
            mag_matrix[:, j] = np.sum(S[idx, :], axis=0)
    np.savetxt('mag_matrix.csv', mag_matrix, delimiter=',')

    with open('audio.pkl','wb') as f:
        pickle.dump((notes, mag_matrix.T, times), f)

    # # Plot the magnitude matrix
    # plt.figure(figsize=(10, 6))
    # plt.imshow(mag_matrix.T, aspect='auto', origin='lower', cmap='viridis', extent=[times[0], times[-1], 0, len(notes)])
    # plt.colorbar(label='Magnitude')
    # plt.yticks(ticks=np.arange(len(notes)), labels=notes)
    # plt.xlabel('Time (s)')
    # plt.ylabel('Notes')
    # plt.title('Magnitude Matrix')
    # plt.tight_layout()
    # plt.show()
    print(mag_matrix)
    return mag_matrix