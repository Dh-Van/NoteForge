# test.py

import pickle
import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt
import scipy.signal as sg
from utils import note_to_freq

# --- Envelope‐extraction parameters & functions ---
DILATION = (3, 5)

def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    lmin = (np.diff(np.sign(np.diff(s))) >= 0).nonzero()[0] + 1
    lmax = (np.diff(np.sign(np.diff(s))) <= 0).nonzero()[0] + 1
    if split:
        mid = np.mean(s)
        lmin = lmin[s[lmin] < mid]
        lmax = lmax[s[lmax] > mid]
    lmin = lmin[[i + np.argmin(s[lmin[i:i + dmin]]) for i in range(0, len(lmin), dmin)]]
    lmax = lmax[[i + np.argmax(s[lmax[i:i + dmax]]) for i in range(0, len(lmax), dmax)]]
    return lmin, lmax


def compute_envelope(data, samplerate, cutoff,
                     n_chunks=50, pad_chunks=1, max_iter=200):
    low, high = cutoff
    sos      = sg.butter(5, (low, high), fs=samplerate, btype='bandpass', output='sos')
    filtered = sg.sosfilt(sos, data)
    rect     = np.abs(filtered)

    chunk_size  = int(np.ceil(len(rect) / n_chunks))
    rect_chunks = np.array([
        rect[i*chunk_size : min((i+1)*chunk_size, len(rect))].max()
        for i in range(int(np.ceil(len(rect)/chunk_size)))
    ])

    padded = np.pad(rect_chunks, pad_chunks, constant_values=0.0)
    M = len(padded)
    x = np.arange(M)

    _, extrema = hl_envelopes_idx(padded)
    idxs = np.where(padded > 0.01 * padded.max())[0]
    A_c, D_c = (idxs[0], idxs[-1]) if idxs.size else (0, M-1)
    seeds = np.sort(np.unique(np.concatenate([extrema, [A_c, D_c]])))

    prev_len = -1
    iters = 0
    while len(seeds) != prev_len and iters < max_iter:
        prev_len = len(seeds)
        interp = np.interp(x, seeds, padded[seeds])
        diff   = padded - interp
        idx    = np.argmax(diff)
        if padded[idx] <= interp[idx]:
            break
        seeds = np.sort(np.append(seeds, idx))
        iters += 1
    absinterp = interp

    fp    = np.diff(absinterp)
    fp_sm = np.convolve(fp, np.ones(DILATION[0]), mode='same')
    fpp   = np.diff(fp_sm)
    fpp_sm= np.convolve(fpp, np.ones(DILATION[1]), mode='same')

    segment = fpp_sm[A_c : D_c+1]
    neg     = (-segment).clip(min=0)
    peaks   = np.where(neg > (neg.max() / 20))[0]
    B_c     = A_c + (peaks[0] if peaks.size else 0)
    peaks2  = np.where(neg > 0)[0]
    C_c     = A_c + (peaks2[-1] if peaks2.size else (D_c - A_c))

    types = [
        'AS' if C_c <= B_c else 'ASR',
        'Dynamic' if (absinterp[C_c] - absinterp[B_c]) < 0 else 'Static'
    ]

    A = max((A_c - pad_chunks) * chunk_size, 0)
    B = max((B_c - pad_chunks) * chunk_size, 0)
    C = max((C_c - pad_chunks) * chunk_size, 0)
    D = min((D_c - pad_chunks) * chunk_size, len(data)-1)
    indices = np.array([A, B, C, D])

    env_chunks = absinterp[pad_chunks : M - pad_chunks]
    positions  = np.linspace(0, len(data)-1, num=env_chunks.size)
    env_curve  = np.interp(np.arange(len(data)), positions, env_chunks)

    return indices, types, env_curve

# --- Loaders & Helpers ---

def load_reference_db(path='reference_db.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_audio_data(path='audio.pkl'):
    with open(path, 'rb') as f:
        notes, mag_T, times = pickle.load(f)
    return list(notes), np.asarray(mag_T).T, np.asarray(times)


def load_audio_file(path):
    audio, fs = sf.read(path)
    if audio.ndim > 1:
        audio = audio[:,0]
    return audio, fs


def find_segments(mask):
    edges  = np.diff(mask.astype(int))
    starts = np.where(edges == 1)[0] + 1
    ends   = np.where(edges == -1)[0] + 1
    if mask[0]: starts = np.insert(starts, 0, 0)
    if mask[-1]: ends = np.append(ends, len(mask))
    return list(zip(starts, ends))

# --- Scoring & Synthesis (attack & release matching) ---

def score_events(mag_matrix, reference_db, note_list,
                 audio, fs, times,
                 thr_ratio=0.02, corr_thresh=0.0):
    events = []
    for i, note in enumerate(note_list):
        mag = mag_matrix[:,i]
        if mag.max() == 0:
            continue
        thr = max(thr_ratio * mag.max(), 1e-6)
        for s_f, e_f in find_segments(mag >= thr):
            i0 = int(times[s_f] * fs)
            i1 = int(min(len(audio), np.ceil(times[e_f] * fs)))
            segment = audio[i0:i1]

            # compute its envelope and indices
            idxs, types, env_curve = compute_envelope(segment, fs,
                (note_to_freq(note[:-1], note[-1]) / np.sqrt(2),
                 note_to_freq(note[:-1], note[-1]) * np.sqrt(2)))
            A, B, C, D = idxs
            seg_attack  = env_curve[A:B]
            seg_release = env_curve[C:D]

            best_inst, best_score = None, -1.0
            for (inst, ref_note), suls in reference_db.items():
                if ref_note != note: continue
                for info in suls.values():
                    ref_attack  = info['attack']
                    ref_release = info['release']
                    # correlate attack
                    L_a = min(len(seg_attack), len(ref_attack))
                    if L_a < 3: continue
                    ca = np.corrcoef(seg_attack[:L_a], ref_attack[:L_a])[0,1]
                    # correlate release
                    L_r = min(len(seg_release), len(ref_release))
                    if L_r < 3: continue
                    cr = np.corrcoef(seg_release[:L_r], ref_release[:L_r])[0,1]
                    score = (ca + cr) / 2
                    if score > best_score:
                        best_inst, best_score = inst, score

            if best_inst and best_score >= corr_thresh:
                events.append((note, s_f, e_f, best_inst))
    return events


def synthesize_instrument(audio_file, reference_db, instrument,
                          audio_data_path='audio.pkl',
                          thr_ratio=0.1, corr_thresh=0.05,
                          safety_semitones=4):
    notes, mag_matrix, times = load_audio_data(audio_data_path)
    audio, fs = load_audio_file(audio_file)
    events = score_events(mag_matrix, reference_db, notes,
                          audio, fs, times,
                          thr_ratio, corr_thresh)
    print(f"[synth] {len(events)} events detected for '{instrument}'")

    separated = np.zeros_like(audio)
    for note, s_f, e_f, inst in events:
        if inst.lower() != instrument.lower():
            continue
        i0 = int(times[s_f] * fs)
        i1 = int(min(len(audio), np.ceil(times[e_f] * fs)))
        segment = audio[i0:i1]

        tone, octv = note[:-1], note[-1]
        f0 = note_to_freq(tone, octv)
        low  = f0 * 2**(-safety_semitones/12)
        high = f0 * 2**( safety_semitones/12)
        b, a = butter(4, [low/(fs/2), high/(fs/2)], btype='band')
        filtered = filtfilt(b, a, segment)

        _, types, env_curve = compute_envelope(filtered, fs, (low, high))
        # match envelope length
        if len(env_curve) != len(filtered):
            env_curve = np.interp(
                np.arange(len(filtered)),
                np.linspace(0, len(filtered)-1, num=len(env_curve)),
                env_curve
            )
        separated[i0:i1] += filtered * env_curve

    peak = np.max(np.abs(separated))
    if peak > 0:
        separated /= peak
    else:
        print(f"[synth] Warning: output silent for '{instrument}'")

    return separated, fs


def save_wave(path, waveform, fs):
    sf.write(path, waveform, fs)


if __name__ == "__main__":
    ref_db = load_reference_db('reference_db.pkl')
    synth, fs = synthesize_instrument(
        audio_file='output/SugarPlum/SugarPlum_mono.wav',
        reference_db=ref_db,
        instrument='clarinet'
    )
    save_wave('clarinet_separated.wav', synth, fs)
    print("Done! 🎶")
