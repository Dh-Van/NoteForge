# generate_reference_db.py

import os
import pickle
from io import BytesIO
from zipfile import ZipFile

import requests
from pydub import AudioSegment
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

from utils import bandpass_filter, extract_envelope, segment_envelope, note_to_freq, normalize_envelope

# ===================================================================
# Instrument sample URLs 
# ===================================================================
INSTRUMENT_URLS = [
    ("Flute_V",    "https://theremin.music.uiowa.edu/sound%20files/MIS%20Pitches%20-%202014/Woodwinds/Flute/Flute.vib.ff.stereo.zip"),
    ("Flute_N",    "https://theremin.music.uiowa.edu/sound%20files/MIS%20Pitches%20-%202014/Woodwinds/Flute/Flute.nonvib.ff.stereo.zip"),
    ("Clarinet_b", "https://theremin.music.uiowa.edu/sound%20files/MIS%20Pitches%20-%202014/Woodwinds/Bb%20Clarinet/BbClarinet.ff.stereo.zip"),
]

def get_single_instrument_sample(inst_tag: str, url: str):
    """
    Download & unpack AIFF samples for one instrument tag,
    convert to WAV under instrument_samples/<family>/<inst_tag>/.
    """
    family = inst_tag.split("_")[0]  # e.g. "Flute" or "Clarinet"
    dest_dir = os.path.join("instrument_samples", family, inst_tag)
    os.makedirs(dest_dir, exist_ok=True)

    # Skip if samples already exist
    if os.path.exists(dest_dir) and len(os.listdir(dest_dir)) > 0:
        print(f"[generate_db] Samples for {inst_tag} already exist in {dest_dir}")
        return

    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"[generate_db] ERROR downloading {inst_tag}: HTTP {resp.status_code}")
        return

    with ZipFile(BytesIO(resp.content)) as zf:
        for name in zf.namelist():
            if name.startswith("__MACOSX/") or not name.lower().endswith(".aif"):
                continue
            raw = BytesIO(zf.read(name))
            audio = AudioSegment.from_file(raw, format="aiff")
            parts = name.rsplit(".", maxsplit=3)
            if len(parts) < 3:
                continue
            base, sul, _fmt = parts[0], parts[1], parts[2]
            note = base.split(".")[-1]
            out_name = f"{family}.{sul}.{note}.wav"
            out_path = os.path.join(dest_dir, out_name)
            audio.export(out_path, format="wav")
    print(f"[generate_db] Samples for {inst_tag} saved to {dest_dir}")

def process_instrument_sample(db, wav_path, family, sul, note):
    """
    Process a single instrument sample and add to the reference database.
    
    Args:
        db: Reference database dict
        wav_path: Path to WAV file
        family: Instrument family (e.g., "Flute", "Clarinet")
        sul: Playing style (e.g., "vib", "nonvib")
        note: Note name (e.g., "A4")
    """
    # Load the audio
    waveform, sr = sf.read(wav_path)
    if waveform.ndim > 1:
        waveform = waveform[:, 0]  # Convert to mono
    
    # Normalize
    if np.max(np.abs(waveform)) > 0:
        waveform = waveform / np.max(np.abs(waveform))
    
    # Extract the envelope
    envelope = extract_envelope(waveform, sr)
    
    # Segment the envelope
    segments, env_type = segment_envelope(envelope)
    A, S, R = segments
    
    # Extract attack, sustain, release segments
    attack = envelope[A:S+1] if S > A else np.array([])
    sustain = envelope[S:R+1] if R > S else np.array([])
    release = envelope[R:] if R < len(envelope)-1 else np.array([])
    
    # Normalize segments for comparison
    attack_norm = normalize_envelope(attack, 50) if len(attack) > 0 else np.array([])
    sustain_norm = normalize_envelope(sustain, 50) if len(sustain) > 0 else np.array([])
    release_norm = normalize_envelope(release, 50) if len(release) > 0 else np.array([])
    
    # Store in database
    key = (family.lower(), note.upper())
    db.setdefault(key, {})[sul] = {
        "waveform": waveform,
        "sampling_rate": sr,
        "envelope": envelope,
        "attack": attack,
        "sustain": sustain,
        "release": release,
        "attack_norm": attack_norm,
        "sustain_norm": sustain_norm,
        "release_norm": release_norm,
        "type": env_type,
        "segments": segments
    }
    
    return True

def build_reference_db():
    """
    Walk through instrument_samples/, load each WAV, extract envelopes,
    and populate the reference_db dict.
    
    Returns:
        Reference database dict
    """
    reference_db = {}
    root = "instrument_samples"
    
    if not os.path.exists(root):
        print(f"[generate_db] Error: {root} directory not found")
        return reference_db
    
    for family in os.listdir(root):
        fam_path = os.path.join(root, family)
        if not os.path.isdir(fam_path):
            continue
            
        for inst_tag in os.listdir(fam_path):
            inst_path = os.path.join(fam_path, inst_tag)
            if not os.path.isdir(inst_path):
                continue
                
            for fname in os.listdir(inst_path):
                if not fname.lower().endswith(".wav"):
                    continue
                
                parts = fname.split(".")
                if len(parts) < 3:
                    continue
                    
                sul = parts[1]
                note = parts[2]
                wav_path = os.path.join(inst_path, fname)
                
                try:
                    process_instrument_sample(reference_db, wav_path, family, sul, note)
                except Exception as e:
                    print(f"[generate_db] Error processing {wav_path}: {e}")
    
    print(f"[generate_db] Processed {len(reference_db)} note-instrument combinations")
    return reference_db

def visualize_reference_db(db, output_dir="reference_visualizations"):
    """
    Create visualizations of the reference database envelopes.
    
    Args:
        db: Reference database
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by instrument family
    families = {}
    for (family, note), sul_dict in db.items():
        families.setdefault(family, []).append((note, sul_dict))
    
    # For each family, create a grid of envelope plots
    for family, items in families.items():
        # Sort by note name
        items.sort()
        
        # Create pages of up to 25 plots each
        pages = [items[i:i+25] for i in range(0, len(items), 25)]
        for page_idx, page_items in enumerate(pages):
            n_items = len(page_items)
            rows = int(np.ceil(np.sqrt(n_items)))
            cols = int(np.ceil(n_items / rows))
            
            fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
            # force axes into a (rows x cols) array
            axes = np.array(axes, dtype=object).reshape(rows, cols)
            fig.suptitle(f"{family.capitalize()} Envelopes (Page {page_idx+1})")
            
            # Plot each note's envelopes
            for idx, (note, sul_dict) in enumerate(page_items):
                r, c = divmod(idx, cols)
                ax = axes[r, c]
                for sul, info in sul_dict.items():
                    ax.plot(info['envelope'], label=sul)
                    A, S, R = info['segments']
                    ax.axvline(x=A, color='g', linestyle='--', alpha=0.5)
                    ax.axvline(x=S, color='r', linestyle='--', alpha=0.5)
                    ax.axvline(x=R, color='b', linestyle='--', alpha=0.5)
                ax.set_title(note)
                ax.legend(loc='upper right', fontsize='small')
                ax.set_ylim(0, 1.1)
                ax.set_xlabel("Sample")
                ax.set_ylabel("Amplitude")
            
            # Hide any unused subplots
            for idx in range(n_items, rows * cols):
                r, c = divmod(idx, cols)
                axes[r, c].axis('off')
            
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            output_path = os.path.join(output_dir, f"{family}_envelopes_page{page_idx+1}.png")
            plt.savefig(output_path, dpi=150)
            plt.close(fig)
            print(f"[generate_db] Created visualization: {output_path}")

def main():
    # 1) Download & convert samples for all instruments
    for inst, url in INSTRUMENT_URLS:
        print(f"[generate_db] Fetching samples for {inst}…")
        get_single_instrument_sample(inst, url)

    # 2) Build reference database
    reference_db = build_reference_db()
    
    # 3) Create visualizations
    visualize_reference_db(reference_db)

    # 4) Persist to reference_db.pkl
    with open("reference_db.pkl", "wb") as f:
        pickle.dump(reference_db, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[generate_db] reference_db.pkl created with {len(reference_db)} entries")

if __name__ == "__main__":
    main()
