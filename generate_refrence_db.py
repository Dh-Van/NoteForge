import os
import pickle
from io import BytesIO
from zipfile import ZipFile

import requests
from pydub import AudioSegment
import soundfile as sf
import numpy as np

import utils
from utils import bandpass_filter, extract_envelope, segment_envelope, note_to_freq, normalize_envelope

# ===================================================================
# Instrument sample URLs
# ===================================================================
INSTRUMENT_URLS = [
    ("Flute_V",    "https://theremin.music.uiowa.edu/sound%20files/MIS%20Pitches%20-%202014/Woodwinds/Flute/Flute.vib.ff.stereo.zip"),
    ("Flute_N",    "https://theremin.music.uiowa.edu/sound%20files/MIS%20Pitches%20-%202014/Woodwinds/Flute/Flute.nonvib.ff.stereo.zip"),
    ("Clarinet_b", "https://theremin.music.uiowa.edu/sound%20files/MIS%20Pitches%20-%202014/Woodwinds/Bb%20Clarinet/BbClarinet.ff.stereo.zip"),
]

# ===================================================================
# Download & unpack AIFF samples using custom naming logic
# ===================================================================
def get_single_instrument_sample(inst_tag: str, url: str):
    dest_dir = os.path.join("instrument_samples", inst_tag)
    os.makedirs(dest_dir, exist_ok=True)
    if os.listdir(dest_dir):
        print(f"[generate_db] Samples exist for {inst_tag}")
        return

    resp = requests.get(url)
    resp.raise_for_status()
    with ZipFile(BytesIO(resp.content)) as zf:
        for name in zf.namelist():
            if name.startswith("__MACOSX/") or not name.lower().endswith(".aif"):
                continue
            raw = BytesIO(zf.read(name))
            audio = AudioSegment.from_file(raw, format="aiff")
            base = os.path.basename(name)
            parts = base.split('.')

            if any(s in parts for s in utils.STRINGS):
                family, sul, note = parts[0], parts[3], parts[4]
            elif any(s in parts[0] for s in utils.WOODWINDS):
                family, sul, note = parts[0], parts[1], parts[3]
            elif any(s in parts for s in utils.CLARINET):
                family, sul, note = parts[0][2:], parts[0][:2], parts[2]
            else:
                continue

            wav_name = f"{family.lower()}.{sul}.{note}.wav"
            out_path = os.path.join(dest_dir, wav_name)
            audio.export(out_path, format="wav")

    print(f"[generate_db] Downloaded samples for {inst_tag}")

def get_all_instrument_samples():
    for inst_tag, url in INSTRUMENT_URLS:
        print(f"[generate_db] Fetching {inst_tag}…")
        get_single_instrument_sample(inst_tag, url)
    print("[generate_db] All samples fetched.")

# ===================================================================
# Inline ASR envelope extraction
# ===================================================================
def compute_asr_envelope(wf: np.ndarray, sr: int, center_freq: float):
    filtered = bandpass_filter(wf, sr, center_freq)
    env = extract_envelope(filtered, sr)
    (A, S, R), env_type = segment_envelope(env)
    return env, (A, S, R), env_type

# ===================================================================
# Build reference DB by walking instrument_samples/
# ===================================================================
def build_reference_db():
    reference_db = {}
    root = "instrument_samples"
    if not os.path.isdir(root):
        print("[generate_db] No instrument_samples folder.")
        return reference_db

    for inst_tag in os.listdir(root):
        inst_path = os.path.join(root, inst_tag)
        if not os.path.isdir(inst_path):
            continue

        family = inst_tag.split("_", 1)[0].lower()
        for fname in os.listdir(inst_path):
            if not fname.lower().endswith(".wav"):
                continue

            # now filenames are uniform: family.sul.note.wav
            parts = fname.split('.')
            if len(parts) != 4:
                # skip anything that didn't match
                continue
            _, sul, note, ext = parts
            if ext.lower() != 'wav':
                continue

            wav_path = os.path.join(inst_path, fname)
            try:
                wf, sr = sf.read(wav_path)
                if wf.ndim > 1:
                    wf = wf[:, 0]
                if np.max(np.abs(wf)) > 0:
                    wf = wf / np.max(np.abs(wf))

                f0 = note_to_freq(note)
                env, (A, S, R), env_type = compute_asr_envelope(wf, sr, f0)

                reference_db.setdefault((family, note.upper()), {})[sul] = {
                    "envelope":     env,
                    "attack_norm":  normalize_envelope(env[A:S], 100),
                    "sustain_norm": normalize_envelope(env[S:R], 100),
                    "release_norm": normalize_envelope(env[R:], 100),
                    "segments":     (A, S, R),
                    "type":         env_type,
                }

            except Exception as e:
                print(f"[generate_db] Error processing {wav_path}: {e}")

    print(f"[generate_db] Built ref DB with {len(reference_db)} entries")
    return reference_db

# ===================================================================
# Run download + build + save
# ===================================================================
if __name__ == "__main__":
    get_all_instrument_samples()
    db = build_reference_db()
    with open("reference_db.pkl", "wb") as f:
        pickle.dump(db, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[generate_db] reference_db.pkl created ({len(db)} entries)")
