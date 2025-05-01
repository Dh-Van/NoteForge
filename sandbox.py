import pickle
import numpy as np
import soundfile
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from utils import note_to_freq

# --- Loaders & Helpers ---
def load_reference_db(path='reference_db.pkl'):
    """Load the reference attack/release database."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_audio(path):
    """Load a mono audio file and return (audio, fs)."""
    audio, fs = soundfile.read(path)
    return (audio if audio.ndim == 1 else audio[:,0]), fs

# --- Frame-wise classification ---
def classify_frames(notes, mag_matrix, reference_db,
                    instruments=('flute','clarinet'), silence_thresh=1e-6):
    """
    Assigns to each frame the most likely note for each instrument, or None if silence.
    Returns a list of dicts: [{ 'flute': note_or_None, 'clarinet': note_or_None }, ...]
    """
    n_frames = mag_matrix.shape[0]
    frame_assns = []
    for t in range(n_frames):
        mags = mag_matrix[t]
        if mags.max() < silence_thresh:
            frame_assns.append({inst: None for inst in instruments})
            continue
        nz = np.nonzero(mags)[0]
        present = [notes[i] for i in nz]
        def score(inst, note):
            best = -np.inf
            for (inst_db, ref_note), suls in reference_db.items():
                if inst_db != inst or ref_note != note:
                    continue
                for info in suls.values():
                    atk = info.get('attack', [])
                    rel = info.get('release', [])
                    if len(atk) > 0 and len(rel) > 0:
                        best = max(best, (np.mean(atk) + np.mean(rel)) / 2)
            return best
        if len(present) == 1:
            solo = present[0]
            scs = {inst: score(inst, solo) for inst in instruments}
            pick = max(scs, key=scs.get)
            assn = {inst: (solo if inst == pick else None) for inst in instruments}
        else:
            remaining = present.copy()
            assn = {}
            for inst in instruments:
                scs = {note: score(inst, note) for note in remaining}
                pick = max(scs, key=scs.get)
                assn[inst] = pick
                remaining.remove(pick)
        frame_assns.append(assn)
    return frame_assns

# --- Plot classification map ---
def plot_instrument_map(notes, times, frame_labels,
                        instruments=('flute','clarinet'), min_duration=0.2):
    """
    Plot x = time, y = note, color-coded by instrument.
    Only plots notes held at least min_duration seconds.
    """
    n_frames = len(frame_labels)
    n_notes = len(notes)
    # frame spacing
    dt = times[1] - times[0] if len(times) > 1 else 0
    frame_threshold = int(np.ceil(min_duration / dt)) if dt > 0 else n_frames
    # Prepare matrix: rows=time frames, cols=note indices
    data = np.zeros((n_frames, n_notes), dtype=int)
    inst_index = {inst: i+1 for i, inst in enumerate(instruments)}  # 0=silence
    for t, labels in enumerate(frame_labels):
        for inst, note in labels.items():
            if note is not None:
                j = notes.index(note)
                data[t, j] = inst_index[inst]
    # Filter short segments
    for j in range(n_notes):
        for code in inst_index.values():
            mask = (data[:, j] == code).astype(int)
            # find runs
            idx = np.where(mask == 1)[0]
            if idx.size == 0:
                continue
            # group consecutive indices
            splits = np.split(idx, np.where(np.diff(idx) != 1)[0]+1)
            for run in splits:
                if len(run) < frame_threshold:
                    data[run, j] = 0
    # Create colormap and normalization
    cmap = mcolors.ListedColormap(['lightgrey','tab:blue','tab:orange'])
    norm = mcolors.BoundaryNorm([0,1,2,3], cmap.N)
    # Display with time on x-axis and notes on y-axis
    fig, ax = plt.subplots(figsize=(12,6))
    extent = [times[0], times[-1], 0, n_notes]
    ax.imshow(
        data.T,
        aspect='auto',
        origin='lower',
        extent=extent,
        cmap=cmap,
        norm=norm
    )
    # y-axis: notes
    ax.set_yticks(np.arange(0.5, n_notes+0.5))
    ax.set_yticklabels(notes)
    ax.set_ylim(0, n_notes)
    # x-axis: time
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Note')
    # Legend
    patches = [mpatches.Patch(color=cmap.colors[i], label=lbl)
               for i, lbl in enumerate(['Silence'] + list(instruments))]
    ax.legend(handles=patches, bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    plt.show()

# --- Main ---
if __name__ == '__main__':
    # Load data
    audio, fs = load_audio('output/SugarPlum/SugarPlum_mono.wav')
    notes, mag_T, times, fs = pickle.load(open('audio.pkl','rb'))
    mag_matrix = np.asarray(mag_T).T if mag_T.shape[0] == len(notes) else np.asarray(mag_T)
    ref_db = load_reference_db('reference_db.pkl')
    # Classify
    frame_labels = classify_frames(notes, mag_matrix, ref_db)
    # Plot with minimum duration filter
    plot_instrument_map(notes, times, frame_labels, min_duration=0.1)