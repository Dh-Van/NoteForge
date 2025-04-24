# generate_reference_db.py

import os
import pickle
from io import BytesIO
from zipfile import ZipFile

import requests
from pydub import AudioSegment
import soundfile as sf

import utils
from test import compute_envelope  # assumes compute_envelope is defined in test.py

# List of instrument sample URLs
INSTRUMENT_URLS = [
    # ("Flute_V",    "https://theremin.music.uiowa.edu/sound%20files/MIS%20Pitches%20-%202014/Woodwinds/Flute/Flute.vib.ff.stereo.zip"),
    # ("Flute_N",    "https://theremin.music.uiowa.edu/sound%20files/MIS%20Pitches%20-%202014/Woodwinds/Flute/Flute.nonvib.ff.stereo.zip"),
    ("Clarinet_b", "https://theremin.music.uiowa.edu/sound%20files/MIS%20Pitches%20-%202014/Woodwinds/Bb%20Clarinet/BbClarinet.ff.stereo.zip"),
]

def get_single_instrument_sample(instrument, url):
    """
    Downloads and extracts all WAV samples for one instrument.
    Places them under instrument_samples/{family}/{instrument}/
    """
    dest_dir = f'instrument_samples/{instrument[:-2]}/{instrument}'
    os.makedirs(dest_dir, exist_ok=True)

    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"ERROR downloading {instrument}: HTTP {resp.status_code}")
        return

    with ZipFile(BytesIO(resp.content)) as z:
        for name in z.namelist():
            if name.startswith("__MACOSX/"):
                continue
            data = BytesIO(z.read(name))
            # convert AIFF to WAV
            audio = AudioSegment.from_file(data, format="aiff")
            parts = name.split(".")
            # determine filename
            if any(s in parts for s in utils.STRINGS):
                fname = f"{parts[0]}.{parts[3]}.{parts[4]}"
            elif any(s in parts[0] for s in utils.WOODWINDS):
                fname = f"{parts[0]}.{parts[1]}.{parts[3]}"
            elif any(s in parts for s in utils.CLARINET):
                fname = f"{parts[0][2:]}.{parts[0][:2]}.{parts[2]}"
            else:
                continue

            out_path = os.path.join(dest_dir, f"{fname}.wav")
            audio.export(out_path, format="wav")


def get_all_instrument_samples():
    """
    Download and unpack every instrument defined in INSTRUMENT_URLS.
    """
    for inst, url in INSTRUMENT_URLS:
        print(f"[generate_db] Fetching samples for {inst}...")
        get_single_instrument_sample(inst, url)
    print("[generate_db] All instrument samples downloaded.")


def save_single_asr(db, instrument, sul, note, waveform, sr):
    """
    Compute ASR envelope for one note sample and store in db:
      db[(instrument, note)][sul] = { waveform, sr, envelope, attack, sustain, release }
    """
    note_letter, octave = utils.split_note_name(note)
    freq = utils.note_to_freq(note_letter, octave)
    cutoff = (freq / (2**0.5), freq * (2**0.5))

    idx, types, env = compute_envelope(waveform, sr, cutoff)
    A, B, C, D = idx

    key = (instrument.lower(), note.upper())
    db.setdefault(key, {})[sul] = {
        "waveform":     waveform,
        "sampling_rate": sr,
        "envelope":      env,
        "attack":        env[A:B],
        "sustain":       env[B:C],
        "release":       env[C:D],
    }


def save_all_asr(db):
    """
    Walk through instrument_samples/, load each WAV, and extract ASR envelopes.
    """
    root = "instrument_samples"
    for family in os.listdir(root):
        fam_path = os.path.join(root, family)
        for inst in os.listdir(fam_path):
            inst_path = os.path.join(fam_path, inst)
            for fname in os.listdir(inst_path):
                if not fname.lower().endswith(".wav"):
                    continue
                sul = fname.split(".")[1]                # e.g. 'V' or 'N'
                note = fname.split(".")[2]           # e.g. 'Flute.C4'
                wav_path = os.path.join(inst_path, fname)
                wf, sr = sf.read(wav_path)
                if wf.ndim > 1:
                    wf = wf[:, 0]
                save_single_asr(db, family, sul, note, wf, sr)


def create_db():
    """
    1) Build ASR reference DB
    2) Pickle to reference_db.pkl
    """

    # 2) compute and save ASR envelopes
    reference_db = {}
    save_all_asr(reference_db)

    # 3) persist to disk
    with open("reference_db.pkl", "wb") as f:
        pickle.dump(reference_db, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[generate_db] reference_db.pkl created with {len(reference_db)} entries")


# get_all_instrument_samples()
create_db()
