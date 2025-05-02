import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter
from utils import note_to_midi, normalize_envelope

# ─── Classification Parameters ─────────────────────────────
SILENCE_THRESHOLD = 0.01      # Ignore magnitudes below this value
EVENT_MIN_DURATION = 10       # Minimum event length in frames

# Map instrument families to IDs
INSTRUMENT_IDS = {
    'flute':    1,
    'clarinet': 2,
}

# ─── I/O ───────────────────────────────────────────────────
def load_reference_db(path='reference_db.pkl'):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[test] Could not load reference DB: {e}")
        return {}


def load_audio_data(path='audio.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)

# ─── Event segmentation ────────────────────────────────────
def segment_events(mag, threshold=SILENCE_THRESHOLD):
    events = []
    in_event = False
    start = 0
    for i, val in enumerate(mag):
        if not in_event and val > threshold:
            in_event = True
            start = i
        elif in_event and val <= threshold:
            if i - start >= EVENT_MIN_DURATION:
                events.append((start, i))
            in_event = False
    if in_event and len(mag) - start >= EVENT_MIN_DURATION:
        events.append((start, len(mag)))
    return events

# ─── Frame-wise classification ─────────────────────────────
def classify_events_with_envelope_matching(data, ref_db=None):
    notes = data['notes']
    mag   = data['mag_matrix']  # shape: (n_frames, n_notes)
    n_frames, n_notes = mag.shape

    # Define MIDI ranges
    FLUTE_RANGE     = (59, 98)
    CLARINET_RANGE  = (50, 91)

    # Precompute MIDI numbers
    midi_numbers = [note_to_midi(n) for n in notes]

    class_map = np.zeros((n_frames, n_notes), dtype=int)

    for t in range(n_frames):
        # Flute: loudest in range
        flute_candidates = [j for j, m in enumerate(midi_numbers)
                             if FLUTE_RANGE[0] <= m <= FLUTE_RANGE[1]]
        if not flute_candidates:
            flute_candidates = list(range(n_notes))
        j_flute = max(flute_candidates, key=lambda j: mag[t, j])
        class_map[t, j_flute] = INSTRUMENT_IDS['flute']

        # Clarinet: next loudest excluding flute's note
        clarinet_candidates = [j for j, m in enumerate(midi_numbers)
                                if CLARINET_RANGE[0] <= m <= CLARINET_RANGE[1] and j != j_flute]
        if not clarinet_candidates:
            clarinet_candidates = [j for j in range(n_notes) if j != j_flute]
        j_clarinet = max(clarinet_candidates, key=lambda j: mag[t, j])
        class_map[t, j_clarinet] = INSTRUMENT_IDS['clarinet']

    return class_map

# ─── Plotting routines ─────────────────────────────────────
def plot_classification(notes, times, class_map, output_path=None):
    n_frames, n_notes = class_map.shape
    sm = gaussian_filter(class_map.astype(float), sigma=(1.0, 0))
    cmap = mcolors.ListedColormap(['white','tab:blue','tab:green'])
    norm = mcolors.BoundaryNorm([-0.5,0.5,1.5,2.5], cmap.N)

    plt.figure(figsize=(14,10))
    plt.imshow(sm.T, aspect='auto', origin='lower',
               extent=[times[0], times[-1], 0, n_notes],
               cmap=cmap, norm=norm)
    plt.xlabel('Time (s)')
    plt.ylabel('Note')
    plt.title('Instrument Classification')
    plt.yticks(np.arange(n_notes)+0.5, notes)
    legend = [
        mpatches.Patch(color='white',  label='Silence'),
        mpatches.Patch(color='tab:blue', label='Flute'),
        mpatches.Patch(color='tab:green',label='Clarinet'),
    ]
    plt.legend(handles=legend, loc='upper right')
    if output_path:
        plt.tight_layout(); plt.savefig(output_path, dpi=300); plt.close()
        return output_path
    else:
        plt.tight_layout(); plt.show()


def plot_magnitude(notes, times, mag_matrix, output_path=None):
    n_frames, n_notes = mag_matrix.shape
    mag_log = np.log1p(mag_matrix)
    mag_norm = mag_log / mag_log.max() if mag_log.max()>0 else mag_log
    sm = gaussian_filter(mag_norm, sigma=(1.0,0))

    plt.figure(figsize=(14,10))
    plt.imshow(sm.T, aspect='auto', origin='lower',
               extent=[times[0], times[-1], 0, n_notes],
               cmap='viridis')
    plt.xlabel('Time (s)')
    plt.ylabel('Note')
    plt.title('Magnitude Spectrogram')
    plt.yticks(np.arange(n_notes)+0.5, notes)
    if output_path:
        plt.tight_layout(); plt.savefig(output_path, dpi=300); plt.close()
        return output_path
    else:
        plt.tight_layout(); plt.show()


def plot_piano_roll(notes, times, class_map, output_path=None):
    n_frames, n_notes = class_map.shape
    plt.figure(figsize=(14,12))
    colors = {1:'tab:blue',2:'tab:green'}

    for inst in [1, 2]:
        for j in range(n_notes):
            i = 0
            while i < n_frames:
                if class_map[i, j] == inst:
                    start = i
                    while i < n_frames and class_map[i, j] == inst:
                        i += 1
                    end = min(i, n_frames - 1)
                    plt.plot([times[start], times[end]], [j, j], color=colors[inst], linewidth=5)
                else:
                    i += 1

    plt.xlabel('Time (s)')
    plt.ylabel('Note')
    plt.title('Piano Roll')
    plt.yticks(np.arange(n_notes), notes)
    if output_path:
        plt.tight_layout(); plt.savefig(output_path, dpi=300); plt.close()
        return output_path
    else:
        plt.tight_layout(); plt.show()

# ─── Main ───────────────────────────────────────────────────
def main(audio_pkl_path='audio.pkl', ref_db_path='reference_db.pkl'):
    out_dir = os.path.dirname(audio_pkl_path) or 'output'
    os.makedirs(out_dir, exist_ok=True)

    print(f"[test] Loading audio data from {audio_pkl_path}")
    data = load_audio_data(audio_pkl_path)
    print(f"[test] Loading reference DB from {ref_db_path}")
    ref_db = load_reference_db(ref_db_path)

    print("[test] Classifying instruments per frame...")
    class_map = classify_events_with_envelope_matching(data, ref_db)

    print("[test] Plotting classification map...")
    plot_classification(data['notes'], data['times'], class_map,
                        os.path.join(out_dir, 'classification_map.png'))
    print("[test] Plotting magnitude spectrogram...")
    plot_magnitude(data['notes'], data['times'], data['mag_matrix'],
                   os.path.join(out_dir, 'magnitude.png'))
    print("[test] Plotting piano roll...")
    plot_piano_roll(data['notes'], data['times'], class_map,
                    os.path.join(out_dir, 'piano_roll.png'))

    print("[test] Done.")
    return {
        'classification_map': os.path.join(out_dir, 'classification_map.png'),
        'magnitude':          os.path.join(out_dir, 'magnitude.png'),
        'piano_roll':         os.path.join(out_dir, 'piano_roll.png')
    }

if __name__ == '__main__':
    import sys
    ap = sys.argv[1] if len(sys.argv) > 1 else 'audio.pkl'
    dbp = sys.argv[2] if len(sys.argv) > 2 else 'reference_db.pkl'
    main(ap, dbp)
