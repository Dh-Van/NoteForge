import pickle
import numpy as np
import soundfile as sf
import utils
import scipy.signal as sg
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

# --- Envelope extraction functions ---
DILATION = (3, 5)

def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    """Return local minima and maxima indices for envelope seeding."""
    lmin = (np.diff(np.sign(np.diff(s))) >= 0).nonzero()[0] + 1
    lmax = (np.diff(np.sign(np.diff(s))) <= 0).nonzero()[0] + 1
    if split:
        mid = np.mean(s)
        lmin = lmin[s[lmin] < mid]
        lmax = lmax[s[lmax] > mid]
    lmin = lmin[[i + np.argmin(s[lmin[i:i + dmin]]) for i in range(0, len(lmin), dmin)]]
    lmax = lmax[[i + np.argmax(s[lmax[i:i + dmax]]) for i in range(0, len(lmax), dmax)]]
    return lmin, lmax


def compute_envelope(data, samplerate, cutoff, n_chunks=50, pad_chunks=1, max_iter=200):
    """Compute amplitude envelope curve for a segment."""
    low, high = cutoff
    sos = sg.butter(5, (low, high), fs=samplerate, btype='bandpass', output='sos')
    filtered = sg.sosfilt(sos, data)
    rect = np.abs(filtered)
    chunk_size = int(np.ceil(len(rect) / n_chunks))
    rect_chunks = [rect[i*chunk_size : min((i+1)*chunk_size, len(rect))].max()
                   for i in range(int(np.ceil(len(rect)/chunk_size)))]
    rect_chunks = np.array(rect_chunks)
    padded = np.pad(rect_chunks, pad_chunks, constant_values=0.0)
    M, x = len(padded), np.arange(len(padded))
    _, extrema = hl_envelopes_idx(padded)
    idxs = np.where(padded > 0.01 * padded.max())[0]
    A_c, D_c = (idxs[0], idxs[-1]) if idxs.size else (0, M-1)
    seeds = np.sort(np.unique(np.concatenate([extrema, [A_c, D_c]])))
    prev_len = -1
    while len(seeds) != prev_len and max_iter > 0:
        prev_len = len(seeds)
        interp = np.interp(x, seeds, padded[seeds])
        diff = padded - interp
        idx = np.argmax(diff)
        if padded[idx] <= interp[idx]:
            break
        seeds = np.sort(np.append(seeds, idx))
        max_iter -= 1
    absenv = interp
    env_chunks = absenv[pad_chunks:-pad_chunks] if pad_chunks else absenv
    positions = np.linspace(0, len(data)-1, num=env_chunks.size)
    env_curve = np.interp(np.arange(len(data)), positions, env_chunks)
    return env_curve

# --- Data loaders ---

def load_reference_db(path: str) -> dict:
    """Load ASR envelope database."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_audio_data(path: str = 'audio.pkl') -> tuple:
    """Load note_list, magnitude matrix, and frame times."""
    with open(path, 'rb') as f:
        note_list, mag_matrix, times = pickle.load(f)
    return np.asarray(note_list), np.asarray(mag_matrix), np.asarray(times)

# --- Segment utilities ---

def find_segments(mask: np.ndarray) -> list:
    """Find start/end indices of True segments in a boolean mask."""
    edges = np.diff(mask.astype(int))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1
    if mask[0]:
        starts = np.insert(starts, 0, 0)
    if mask[-1]:
        ends = np.append(ends, len(mask))
    return list(zip(starts, ends))

# --- Monophonic labeling for a single instrument ---

def score_events(mag_matrix: np.ndarray,
                 reference_db: dict,
                 note_list: list,
                 thr_ratio: float = 0.05) -> list:
    """Detect and score note segments against each instrument's envelopes."""
    events = []
    for i, note in enumerate(note_list):
        mag = mag_matrix[i]
        if mag.max() == 0:
            continue
        threshold = thr_ratio * mag.max()
        for s, e in find_segments(mag >= threshold):
            seg_len = e - s
            if seg_len < 1:
                continue
            seg = mag[s:e]
            for (inst, ref_note), samples in reference_db.items():
                if ref_note != note:
                    continue
                for data in samples.values():
                    attack = data['attack']
                    release = data.get('release', np.array([]))
                    L = min(len(attack), seg_len)
                    x1 = seg[:L] - seg[:L].mean()
                    y1 = attack[:L] - attack[:L].mean()
                    norm1 = np.linalg.norm(x1) * np.linalg.norm(y1)
                    corr1 = np.dot(x1, y1) / norm1 if norm1 > 0 else 0.0
                    if release.size > 0:
                        Lr = min(len(release), seg_len)
                        x2 = seg[-Lr:] - seg[-Lr:].mean()
                        y2 = release[-Lr:] - release[-Lr:].mean()
                        norm2 = np.linalg.norm(x2) * np.linalg.norm(y2)
                        corr2 = np.dot(x2, y2) / norm2 if norm2 > 0 else corr1
                        score = 0.5 * (corr1 + corr2)
                    else:
                        score = corr1
                    events.append((note, s, e, inst, score))
    return events


def pick_best(events: list) -> list:
    """Choose highest-scoring instrument for each note-segment."""
    best = {}
    for note, s, e, inst, score in events:
        key = (note, s, e)
        if key not in best or score > best[key][1]:
            best[key] = (inst, score)
    return [(n, s, e, inst_s[0]) for (n, s, e), inst_s in best.items()]


def build_monophonic_labels(reference_db: dict,
                            instrument: str,
                            audio_data_path: str = 'audio.pkl',
                            thr_ratio: float = 0.05,
                            min_duration_sec: float = 0.05) -> tuple:
    """
    Return (mono_labels, note_list, times) for one instrument,
    discarding segments shorter than min_duration_sec.
    """
    note_list, mag_matrix, times = load_audio_data(audio_data_path)
    events = score_events(mag_matrix, reference_db, note_list, thr_ratio)
    best = pick_best(events)
    N_frames = mag_matrix.shape[1]
    mono = np.full(N_frames, '', dtype=object)
    for note, s, e, inst in best:
        if inst.lower() != instrument.lower():
            continue
        # check duration
        t0 = times[s]
        t1 = times[min(e-1, len(times)-1)]
        if (t1 - t0) < min_duration_sec:
            continue
        mono[s:e] = note
    return mono, note_list, times

# --- Visualization ---

def plot_monophonic_scatter(mono_labels: np.ndarray,
                            note_list: np.ndarray,
                            times: np.ndarray,
                            instrument: str,
                            max_time: float = 20) -> None:
    """Scatter plot of monophonic note events for a single instrument."""
    note_to_idx = {note: i for i, note in enumerate(note_list)}
    xs, ys = [], []
    for t, note in enumerate(mono_labels):
        time = times[t]
        if note and (max_time is None or time <= max_time):
            xs.append(time)
            ys.append(note_to_idx[note])
    plt.figure(figsize=(10, 6))
    plt.scatter(xs, ys, s=10)
    plt.yticks(np.arange(len(note_list)), note_list)
    plt.xlabel('Time (s)')
    plt.ylabel('Note')
    title = f'Monophonic {instrument.capitalize()} Events Over Time'
    if max_time is not None:
        title += f' (first {max_time:.2f}s)'
    plt.title(title)
    plt.grid(True, axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

# --- Main Execution ---
ref_db = load_reference_db('reference_db.pkl')
instrument = 'clarinet'  # change to 'violin', etc.
mono_labels, note_list, times = build_monophonic_labels(
    ref_db, instrument, 'audio.pkl', thr_ratio=0.7, min_duration_sec=0.07)
plot_monophonic_scatter(mono_labels, note_list, times, instrument)

