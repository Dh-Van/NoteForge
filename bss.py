import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, butter, filtfilt
from utils import note_to_freq
import pickle


def separate(fpath, window_size=254, overlap=None, semitone_bandwidth=0.5):
    """
    Compute magnitude matrix for notes over time via STFT, with a bandpass pre-filter.

    Returns (notes, mag_matrix, times):
      - notes: list of note names length N_notes
      - mag_matrix: np.ndarray shape (N_notes, N_frames)
      - times: np.ndarray length N_frames
    """
    # --- Load audio (mono) ---
    with wave.open(fpath, 'rb') as w:
        fs = w.getframerate()
        raw = w.readframes(w.getnframes())
        dtype = {1: np.int8, 2: np.int16, 4: np.int32}[w.getsampwidth()]
        audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        if w.getnchannels() > 1:
            audio = audio.reshape(-1, w.getnchannels())[:, 0]

    # --- Bandpass filter around musical range (A1–G6) ---
    nyq = fs / 2.0
    low_cut = note_to_freq('A', '1') / nyq      # ~55 Hz
    high_cut = note_to_freq('G', '6') / nyq    # ~1568 Hz
    b, a = butter(4, [low_cut, high_cut], btype='band')
    audio = filtfilt(b, a, audio)

    # --- STFT ---
    if overlap is None:
        overlap = window_size // 2
    freqs, times, Zxx = stft(audio, fs=fs,
                              nperseg=window_size,
                              noverlap=overlap)
    S = np.abs(Zxx)

    # --- Build note list (A1–G6, flats) ---
    notes = [f + '1' for f in ('A','Bb','B')]
    for octv in range(2, 6):
        for f in ('C','Db','D','Eb','E','F','Gb','G','Ab','A','Bb','B'):
            notes.append(f + str(octv))
    notes += [f + '6' for f in ('C','Db','D','Eb','E','F','Gb','G')]

    N_notes = len(notes)
    N_frames = len(times)
    mag_matrix = np.zeros((N_notes, N_frames), dtype=np.float32)

    # --- Sum magnitudes in ± semitone_bandwidth semitone band ---
    for j, nm in enumerate(notes):
        tone, octave = nm[:-1], nm[-1]
        f0 = note_to_freq(tone, octave)
        low = f0 * (2 ** (-0.5/12))
        high = f0 * (2 ** ( 0.5/12))
        idx = np.where((freqs >= low) & (freqs <= high))[0]
        if idx.size:
            mag_matrix[j] = S[idx, :].sum(axis=0)

    save_pickle(notes, mag_matrix, times)  
    # # Plot full scatter
    # plot_mag_scatter(mag_matrix, notes, times, min_mag=0.1, max_time=10.0)
    # Plot top-5 per frame
    plot_top_n_mag_scatter(mag_matrix, notes, times, top_n=5, max_time=10.0)


    return notes, mag_matrix, times


def save_pickle(notes, mag_matrix, times, out='audio.pkl'):
    """Save computed notes, mag_matrix, and times to pickle."""
    with open(out, 'wb') as f:
        pickle.dump((notes, mag_matrix, times), f)


def plot_mag_scatter(mag_matrix, notes, times,
                     min_mag: float = 0.1,
                     max_time: float = None):
    """
    Scatter of magnitude matrix:
      - x axis: time
      - y axis: note index
      - color: magnitude
    Only points with mag >= min_mag and time <= max_time.
    """
    note_idxs, frame_idxs = np.where(mag_matrix >= min_mag)
    xs = times[frame_idxs]
    ys = note_idxs
    cs = mag_matrix[note_idxs, frame_idxs]

    if max_time is not None:
        mask = xs <= max_time
        xs, ys, cs = xs[mask], ys[mask], cs[mask]

    plt.figure(figsize=(10, 6))
    sc = plt.scatter(xs, ys, c=cs, s=5, cmap='viridis')
    plt.colorbar(sc, label='Magnitude')
    plt.yticks(np.arange(len(notes)), notes)
    plt.xlabel('Time (s)')
    plt.ylabel('Note')
    title = 'Magnitude Scatter'
    if max_time is not None:
        title += f' (≤ {max_time:.2f}s)'
    plt.title(title)
    plt.grid(True, axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_top_n_mag_scatter(mag_matrix, notes, times,
                           top_n: int = 1,
                           max_time: float = None):
    """
    Plot only the top_n largest magnitudes at each time frame.
    - mag_matrix: (N_notes × N_frames)
    - notes:      list of length N_notes
    - times:      array of length N_frames
    """
    xs, ys, cs = [], [], []
    N_notes, N_frames = mag_matrix.shape

    for frame in range(N_frames):
        t = times[frame]
        if max_time is not None and t > max_time:
            continue
        mags = mag_matrix[:, frame]
        top_idxs = np.argsort(mags)[-top_n:]
        for ni in top_idxs:
            xs.append(t)
            ys.append(ni)
            cs.append(mags[ni])

    plt.figure(figsize=(10, 6))
    sc = plt.scatter(xs, ys, c=cs, s=20, cmap='viridis')
    plt.colorbar(sc, label='Magnitude')
    plt.yticks(np.arange(len(notes)), notes)
    plt.xlabel('Time (s)')
    plt.ylabel('Note')
    title = f'Top {top_n} Magnitudes per Frame'
    if max_time is not None:
        title += f' (≤ {max_time:.2f}s)'
    plt.title(title)
    plt.grid(True, axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()