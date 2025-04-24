# bss.py

import wave
import numpy as np
from scipy.signal import stft
import pickle
from utils import note_to_freq

def separate(fpath, window_size=1024, overlap=None):
    """
    fpath       : path to a mono .wav file
    window_size : STFT segment length
    overlap     : STFT overlap (defaults to 50%)
    ---
    Computes the magnitude‐vs‐time matrix for each note,
    saves mag_matrix.csv and audio.pkl, and returns the matrix.
    """
    # 1) Read mono audio
    with wave.open(fpath, 'rb') as w:
        fs    = w.getframerate()
        raw   = w.readframes(w.getnframes())
        dtype = {1: np.int8, 2: np.int16, 4: np.int32}[w.getsampwidth()]
        audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        # if multiple channels, take first
        if audio.ndim > 1:
            audio = audio.reshape(-1, w.getnchannels())[:, 0]

    # 2) STFT
    if overlap is None:
        overlap = window_size // 2
    freqs, times, Zxx = stft(audio, fs=fs, nperseg=window_size, noverlap=overlap)
    S = np.abs(Zxx)

    # 3) Build note list (A1–G6, flats only)
    notes = []
    for tone in ["A", "Bb", "B"]:
        notes.append(f"{tone}1")
    for octave in range(2, 6):
        for tone in ["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"]:
            notes.append(f"{tone}{octave}")
    for tone in ["C","Db","D","Eb","E","F","Gb","G"]:
        notes.append(f"{tone}6")

    # 4) Populate magnitude matrix: frames × notes
    n_frames    = S.shape[1]
    n_notes     = len(notes)
    mag_matrix  = np.zeros((n_frames, n_notes), dtype=np.float32)

    for j, nm in enumerate(notes):
        tone   = nm[:-1]
        octave = nm[-1]
        f0     = note_to_freq(tone, octave)
        # half‐semitone band around each note
        f_low  = f0 * 2 ** (-1/24)
        f_high = f0 * 2 ** ( 1/24)
        idx    = np.where((freqs >= f_low) & (freqs <= f_high))[0]
        if idx.size:
            mag_matrix[:, j] = S[idx, :].sum(axis=0)

    # 5) Save for downstream steps
    with open('audio.pkl', 'wb') as f:
        # note order, transposed mag matrix, and time vector
        pickle.dump((notes, mag_matrix.T, times), f)

    print(f"[bss] Computed mag_matrix with shape {mag_matrix.shape}")
    return mag_matrix
