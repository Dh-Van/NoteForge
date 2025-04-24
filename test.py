import pickle
import numpy as np
import soundfile as sf
import utils
import scipy.signal as sg
from scipy.signal import butter, filtfilt

# --- Envelope extraction parameters & functions ---
SecondDerivThreshold = 0.02
DILATION = (3, 5)

def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    '''Returns local minima and maxima indices for envelope seeding.'''
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
    '''Compute envelope and ASR boundaries. Returns (indices, types, env_curve).'''
    low, high = cutoff
    # 1) Bandpass & rectify
    sos = sg.butter(5, (low, high), fs=samplerate, btype='bandpass', output='sos')
    filtered = sg.sosfilt(sos, data)
    rect = np.abs(filtered)
    # 2) Chunk-wise max
    chunk_size = int(np.ceil(len(rect) / n_chunks))
    rect_chunks = [rect[i*chunk_size : min((i+1)*chunk_size, len(rect))].max()
                   for i in range(int(np.ceil(len(rect)/chunk_size)))]
    rect_chunks = np.array(rect_chunks)
    # 3) Pad
    padded = np.pad(rect_chunks, pad_chunks, constant_values=0.0)
    M, x = len(padded), np.arange(len(padded))
    # 4) Seeds
    _, extrema = hl_envelopes_idx(padded, dmin=1, dmax=1)
    idxs = np.where(padded > 0.01 * padded.max())[0]
    if idxs.size:
        A_c, D_c = idxs[0], idxs[-1]
    else:
        A_c, D_c = 0, len(padded)-1
    seeds = np.sort(np.unique(np.concatenate([extrema, [A_c, D_c]])))
    # 5) Fit upper envelope
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
    absinterp = interp
    # 6) Second derivatives
    fp = np.diff(absinterp)
    fp_sm = np.convolve(fp, np.ones(DILATION[0]), mode='same')
    fpp = np.diff(fp_sm)
    fpp_sm = np.convolve(fpp, np.ones(DILATION[1]), mode='same')
    # 7) ASR chunk boundaries
    seg = fpp_sm[A_c:D_c+1]
    neg = (-seg).clip(min=0)
    peaks = np.where(neg > (neg.max()/20))[0]
    B_c = A_c + (peaks[0] if peaks.size else 0)
    peaks2 = np.where(neg > 0)[0]
    C_c = A_c + (peaks2[-1] if peaks2.size else (D_c - A_c))
    # 8) Types
    types = ['AS' if C_c <= B_c else 'ASR',
             'Dynamic' if (absinterp[C_c] - absinterp[B_c]) < 0 else 'Static']
    # 9) Map to samples
    A = max((A_c - pad_chunks) * chunk_size, 0)
    B = max((B_c - pad_chunks) * chunk_size, 0)
    C = max((C_c - pad_chunks) * chunk_size, 0)
    D = min((D_c - pad_chunks) * chunk_size, len(data)-1)
    indices = np.array([A, B, C, D])
    # 10) Upsample
    env_chunks = absinterp[pad_chunks:-pad_chunks] if pad_chunks else absinterp
    positions = np.linspace(0, len(data)-1, num=env_chunks.size)
    env_curve = np.interp(np.arange(len(data)), positions, env_chunks)
    return indices, types, env_curve

# --- Utility functions ---

def load_reference_db(path: str) -> dict:
    '''Load the reference database.'''  
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_audio_data(path: str = 'audio.pkl') -> tuple:
    '''Load note_list, mag_matrix, times.'''  
    with open(path, 'rb') as f:
        note_list, mag_matrix, times = pickle.load(f)
    return np.asarray(note_list), np.asarray(mag_matrix), np.asarray(times)


def load_audio_file(path: str) -> tuple:
    '''Read mono audio and samplerate.'''  
    audio, fs = sf.read(path)
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio, fs


def find_segments(mask: np.ndarray) -> list:
    '''Find start/end index pairs for True mask.'''  
    edges = np.diff(mask.astype(int))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1
    if mask[0]: starts = np.insert(starts, 0, 0)
    if mask[-1]: ends = np.append(ends, len(mask))
    return list(zip(starts, ends))


def score_events(mag_matrix: np.ndarray, reference_db: dict, note_list: list,
                 thr_ratio: float = 0.05) -> list:
    '''Detect note segments above threshold and assign to instruments.'''
    events = []
    for i, note in enumerate(note_list):
        mag = mag_matrix[i]
        if mag.max() == 0:
            continue
        thr = thr_ratio * mag.max()
        for s_frame, e_frame in find_segments(mag >= thr):
            # any envelope match could be applied here if desired
            for (inst, ref_note), suls in reference_db.items():
                if ref_note == note:
                    events.append((note, s_frame, e_frame, inst))
    return events

def pick_best(events: list) -> list:
    '''Filter to unique note-segment/instrument tuples.'''
    unique = {}
    for note, s, e, inst in events:
        unique[(s, e, inst)] = (note, s, e, inst)
    return list(unique.values())

def synthesize_instrument(audio_file: str, reference_db: dict, instrument: str,
                          audio_data_path: str = 'audio.pkl', thr_ratio: float = 0.05,
                          safety_semitones: int = 6) -> tuple:
    '''Separate an instrument by mask, bandpass filter, and envelope shaping.'''
    note_list, mag_matrix, times = load_audio_data(audio_data_path)
    events = score_events(mag_matrix, reference_db, note_list, thr_ratio)
    best_events = pick_best(events)

    audio, fs = load_audio_file(audio_file)
    separated = np.zeros_like(audio)

    # build frame mask
    frame_mask = np.zeros(len(times), dtype=bool)
    for note, s_f, e_f, inst in best_events:
        if inst.lower() == instrument.lower():
            frame_mask[s_f:e_f] = True
    # fill tiny frame gaps
    for g0, g1 in find_segments(~frame_mask):
        if g1 - g0 <= 1:
            frame_mask[g0:g1] = True

    # convert to sample mask
    sample_mask = np.zeros(len(audio), dtype=bool)
    for fs_f, fe_f in find_segments(frame_mask):
        i0 = int(times[fs_f] * fs)
        i1 = int(min(len(audio), np.ceil(times[min(fe_f,len(times)-1)] * fs)))
        sample_mask[i0:i1] = True

    # process each event
    for note, s_f, e_f, inst in best_events:
        if inst.lower() != instrument.lower():
            continue
        i0 = int(times[s_f] * fs)
        i1 = int(min(len(audio), np.ceil(times[min(e_f,len(times)-1)] * fs)))
        segment = audio[i0:i1]

        # bandpass around note
        tone, octv = note[:-1], note[-1]
        f0 = utils.note_to_freq(tone, octv)
        low = f0 * 2**(-safety_semitones/12)
        high = f0 * 2**(safety_semitones/12)
        b, a = butter(4, [low/(fs/2), high/(fs/2)], btype='band')
        filtered = filtfilt(b, a, segment)

        # shape with computed envelope
        _, _, env_curve = compute_envelope(filtered, fs, cutoff=(low, high))
        shaped = filtered * env_curve

        # apply mask
        separated[i0:i1] += shaped * sample_mask[i0:i1]

    # normalize
    peak = np.max(np.abs(separated))
    if peak > 0:
        separated /= peak
    return separated, fs


def save_wave(path: str, waveform: np.ndarray, fs: int):
    '''Write waveform to WAV.'''
    sf.write(path, waveform, fs)

ref_db = load_reference_db('reference_db.pkl')

synth, fs = synthesize_instrument(
    audio_file='output/SugarPlum/SugarPlum_mono.wav',
    reference_db=ref_db,
    instrument='clarinet',
    audio_data_path='audio.pkl',
    thr_ratio=0.15
)

save_wave('flute_synthesized.wav', synth, fs)

print("Done! 🎶")