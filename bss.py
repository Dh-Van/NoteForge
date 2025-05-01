# bss.py

import wave
import numpy as np
import pickle
import os
from scipy.signal import stft
import matplotlib.pyplot as plt
from utils import note_to_freq, bandpass_filter, extract_envelope, segment_envelope

def separate(
    mono_wav_path: str,
    window_size: int = 1024,
    overlap: int | None = None,
    top_k: int = 5,
    output_dir: str = None
):
    """
    Reads a mono WAV file, computes STFT, keeps only the top_k pitches per frame,
    and saves the results for classification.
    
    Args:
        mono_wav_path: Path to mono WAV file
        window_size: STFT window size
        overlap: STFT overlap (default: window_size // 2)
        top_k: Number of top frequencies to keep per frame
        output_dir: Directory to save outputs (defaults to song name in output/)
    
    Returns:
        Path to the saved audio.pkl file
    """
    # Determine output directory
    if output_dir is None:
        base_name = os.path.basename(mono_wav_path).rsplit('.', 1)[0]
        output_dir = os.path.join("output", base_name)
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "audio.pkl")
    
    print(f"[bss] Processing {mono_wav_path}...")
    
    # 1) load mono audio
    with wave.open(mono_wav_path, 'rb') as w:
        fs      = w.getframerate()
        raw     = w.readframes(w.getnframes())
        sampw   = w.getsampwidth()
        nch     = w.getnchannels()
        dtype   = {1: np.int8, 2: np.int16, 4: np.int32}[sampw]
        audio   = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        if nch > 1:
            audio = audio.reshape(-1, nch).mean(axis=1)
    
    # Normalize audio
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    
    # 2) STFT
    if overlap is None:
        overlap = window_size // 2
    freqs, times, Zxx = stft(audio, fs=fs,
                             nperseg=window_size,
                             noverlap=overlap)
    S = np.abs(Zxx)
    
    # 3) build full note list (A1…C7)
    notes = []
    for tone in ["A", "Bb", "B"]:
        notes.append(f"{tone}1")
    for octave in range(2, 7):
        for tone in ["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"]:
            notes.append(f"{tone}{octave}")
    notes.append("C7")
    
    # 4) sample magnitude at each fundamental
    n_frames = S.shape[1]
    n_notes = len(notes)
    mag_matrix = np.zeros((n_frames, n_notes), dtype=np.float32)
    
    # Process each note
    for j, nm in enumerate(notes):
        f0 = note_to_freq(nm)
        lo, hi = f0 * 2**(-1/24), f0 * 2**(1/24)
        idx = np.where((freqs >= lo) & (freqs <= hi))[0]
        if idx.size:
            mag_matrix[:, j] = S[idx, :].sum(axis=0)
    
    # 5) keep only top_k per frame
    if 1 <= top_k < n_notes:
        top_idxs = np.argpartition(-mag_matrix, top_k-1, axis=1)[:, :top_k]
        filt = np.zeros_like(mag_matrix)
        for t, cols in enumerate(top_idxs):
            filt[t, cols] = mag_matrix[t, cols]
        mag_matrix = filt
    
    # 6) Create envelopes for each note
    note_envelopes = {}
    for j, note in enumerate(notes):
        # Check if this note has any significant energy
        if np.max(mag_matrix[:, j]) > 0.01 * np.max(mag_matrix):
            # Filter the audio around this note's frequency
            f0 = note_to_freq(note)
            filtered = bandpass_filter(audio, fs, f0)
            
            # Get the envelope
            envelope = extract_envelope(filtered, fs)
            
            # Segment and save if significant
            if np.max(envelope) > 0.01:
                segments, env_type = segment_envelope(envelope)
                if env_type != "no_sound":
                    note_envelopes[note] = {
                        "envelope": envelope,
                        "segments": segments,
                        "type": env_type
                    }
    
    # 7) save everything needed for classification
    data = {
        "notes": notes,
        "mag_matrix": mag_matrix,
        "times": times,
        "fs": fs,
        "freqs": freqs,
        "S": S,
        "envelopes": note_envelopes,
        "audio": audio
    }
    
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
    
    # 8) Create and save a spectrogram visualization
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(times, freqs, 10 * np.log10(S + 1e-10), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectrogram')
    plt.colorbar(label='Magnitude [dB]')
    plt.ylim(0, 5000)  # Limit to audible frequency range
    
    spec_path = os.path.join(output_dir, "spectrogram.png")
    plt.savefig(spec_path)
    plt.close()
    
    print(f"[bss] audio.pkl saved to {output_path} ({n_notes} notes × {n_frames} frames, top_k={top_k})")
    print(f"[bss] Found {len(note_envelopes)} active notes with significant energy")
    return output_path