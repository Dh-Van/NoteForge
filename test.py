import pickle
import numpy as np
import soundfile as sf
import utils
from scipy.signal import butter, filtfilt

with open('reference_db.pkl', 'rb') as f:
    reference_db = pickle.load(f)

with open('audio.pkl', 'rb') as f:
    note_list, mag_matrix, times = pickle.load(f)

mag_matrix = np.asarray(mag_matrix)
note_list  = utils.generate_note_list()[:mag_matrix.shape[0]]

def find_segments(mask):
    edges  = np.diff(mask.astype(int))
    starts = np.where(edges ==  1)[0] + 1
    ends   = np.where(edges == -1)[0] + 1
    if mask[0]:
        starts = np.insert(starts, 0, 0)
    if mask[-1]:
        ends = np.append(ends, len(mask))
    return list(zip(starts, ends))

def score_events(mag_matrix, reference_db, note_list, thr_ratio=0.05):
    events = []
    for i, note in enumerate(note_list):
        mag = mag_matrix[i]
        if mag.max() == 0:
            continue
        threshold = thr_ratio * mag.max()
        segments  = find_segments(mag >= threshold)

        for (inst, ref_note), suls in reference_db.items():
            if ref_note != note:
                continue
            for sul, data in suls.items():
                att = data["attack"]
                rel = data.get("release", np.array([]))
                for s_frame, e_frame in segments:
                    seg_len = e_frame - s_frame
                    if seg_len < 1:
                        continue
                    seg = mag[s_frame:e_frame]

                    # attack correlation
                    L_att = min(len(att), seg_len)
                    if L_att < 1:
                        continue
                    x1 = seg[:L_att] - seg[:L_att].mean()
                    y1 = att[:L_att] - att[:L_att].mean()
                    norm1 = np.linalg.norm(x1) * np.linalg.norm(y1)
                    corr1 = np.dot(x1, y1) / norm1 if norm1 > 0 else 0.0

                    if len(rel) >= 1:
                        L_rel = min(len(rel), seg_len)
                        x2 = seg[-L_rel:] - seg[-L_rel:].mean()
                        y2 = rel[-L_rel:] - rel[-L_rel:].mean()
                        norm2 = np.linalg.norm(x2) * np.linalg.norm(y2)
                        corr2 = np.dot(x2, y2) / norm2 if norm2 > 0 else corr1
                        score = 0.5 * (corr1 + corr2)
                    else:
                        score = corr1

                    events.append((note, s_frame, e_frame, inst, score))
    return events

def pick_best(events):
    best = {}
    for note, s, e, inst, score in events:
        key = (note, s, e)
        if key not in best or score > best[key][1]:
            best[key] = (inst, score)
    return [(note, s, e, inst_score[0]) for (note, s, e), inst_score in best.items()]

events       = score_events(mag_matrix, reference_db, note_list, thr_ratio=0.05)
best_events  = pick_best(events)
violin_evts  = [(s, e, note) for (note, s, e, inst) in best_events if inst.lower() == "violin"]

num_frames = mag_matrix.shape[1]
mask       = np.zeros(num_frames, dtype=bool)
for s, e, _ in violin_evts:
    mask[s:e] = True

for gs, ge in find_segments(~mask):
    if ge - gs < 10:
        mask[gs:ge] = True

audio, fs = sf.read('./output/SwampThang/SwampThang_mono.wav')
if audio.ndim > 1:
    audio = audio[:,0]

violin_only = np.zeros_like(audio)
for s_frame, e_frame, note in violin_evts:
    t0 = times[s_frame]
    t1 = times[min(e_frame-1, len(times)-1)]
    i0 = int(max(0, t0 * fs))
    i1 = int(min(len(audio), t1 * fs))
    segment = audio[i0:i1]
    if segment.size == 0:
        continue

    tone, octave = note[:-1], note[-1]
    f0            = utils.note_to_freq(tone, octave)
    low  = f0 * 2 ** (-6/12)
    high = f0 * 2 ** ( 6/12)
    nyq  = fs / 2.0

    b, a = butter(2, [low/nyq, high/nyq], btype='band')
    filtered = filtfilt(b, a, segment)

    violin_only[i0:i1] = filtered

sf.write('violin_attack_release_filtered.wav', violin_only, fs)
print(f"Saved violin_attack_release_filtered.wav ({len(violin_only)/fs:.2f}s)")
