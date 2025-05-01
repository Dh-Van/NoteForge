# utils.py

import re
import numpy as np
from scipy.signal import butter, filtfilt, hilbert

# semitone ↔ name maps
_SEMITONE_MAP = {
    "C": 0,  "C#": 1, "Db": 1,
    "D": 2,  "D#": 3, "Eb": 3,
    "E": 4,
    "F": 5,  "F#": 6, "Gb": 6,
    "G": 7,  "G#": 8, "Ab": 8,
    "A": 9,  "A#": 10,"Bb": 10,
    "B": 11
}
_REVERSE_SEMITONE = {v: k for k, v in _SEMITONE_MAP.items()}

def split_note_name(full_note: str) -> tuple[str, int]:
    """Split 'Bb4' → ('Bb', 4)."""
    m = re.match(r'^([A-G][b#]?)(\d+)$', full_note)
    if not m:
        raise ValueError(f"Cannot parse note: {full_note}")
    note, octs = m.groups()
    return note, int(octs)

def note_to_midi(full_note: str) -> int:
    """Convert 'A4' → 69 (MIDI number)."""
    note, octave = split_note_name(full_note)
    sem = _SEMITONE_MAP[note]
    return 12 * (octave + 1) + sem

def midi_to_note(midi: int) -> str:
    """Convert MIDI number → e.g. 'A4'."""
    octave = (midi // 12) - 1
    sem = midi % 12
    name = _REVERSE_SEMITONE[sem]
    return f"{name}{octave}"

def note_to_freq(full_note: str) -> float:
    """Convert 'A4' → 440.0 Hz."""
    midi = note_to_midi(full_note)
    return 440.0 * 2 ** ((midi - 69) / 12)

def bandpass_filter(signal, fs, center_freq, bandwidth_factor=2**0.5):
    """
    Apply bandpass filter around center_freq with bandwidth determined by bandwidth_factor.
    
    Args:
        signal: Input signal
        fs: Sampling rate
        center_freq: Center frequency
        bandwidth_factor: The bandwidth is center_freq/bandwidth_factor to center_freq*bandwidth_factor
        
    Returns:
        Filtered signal
    """
    nyq = fs / 2.0
    low = (center_freq / bandwidth_factor) / nyq
    high = (center_freq * bandwidth_factor) / nyq
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, signal)

def extract_envelope(signal, fs, hop_size=256):
    """
    Extract the amplitude envelope of a signal.
    
    Args:
        signal: Input signal
        fs: Sampling rate
        hop_size: Size of window for extracting envelope
        
    Returns:
        Envelope signal
    """
    # Calculate envelope using Hilbert transform
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    
    # Further smooth the envelope
    hop_size = min(hop_size, len(amplitude_envelope))
    num_chunks = len(amplitude_envelope) // hop_size
    smoothed_envelope = np.zeros(num_chunks)
    
    for i in range(num_chunks):
        start = i * hop_size
        end = start + hop_size
        smoothed_envelope[i] = np.max(amplitude_envelope[start:end])
    
    # Resample to original size using linear interpolation
    envelope_time = np.linspace(0, 1, num_chunks)
    signal_time = np.linspace(0, 1, len(signal))
    smoothed_envelope = np.interp(signal_time, envelope_time, smoothed_envelope)
    
    return smoothed_envelope

def segment_envelope(envelope, threshold=0.1):
    """
    Segment an envelope into attack, sustain, release parts.
    
    Args:
        envelope: Input amplitude envelope
        threshold: Threshold for determining start/end points
        
    Returns:
        (A, S, R) indices and segment types
    """
    # Normalize envelope
    if np.max(envelope) > 0:
        env = envelope / np.max(envelope)
    else:
        return (0, 0, len(envelope)), "no_sound"
    
    # Find attack start (A)
    above_threshold = np.where(env > threshold)[0]
    if len(above_threshold) == 0:
        return (0, 0, len(envelope)), "no_sound"
    
    A = above_threshold[0]
    
    # Find release end (R)
    R = above_threshold[-1]
    
    # Determine sustain point (S) - approximately where attack ends
    # Look for the first peak after A
    window_size = min(int((R - A) * 0.2) + 1, len(env) - A)
    if window_size <= 1:
        S = A
    else:
        peak_idx = A + np.argmax(env[A:A+window_size])
        S = peak_idx
    
    # Determine envelope type
    if S == A:
        env_type = "AR" # Attack-Release, no sustain
    else:
        # Check if it's exponential decay or sustained
        decay_rate = np.mean(np.diff(env[S:R+1]))
        if decay_rate < -0.01:  # Significant decay
            env_type = "ASR_decay"  # Attack-Sustain-Release with decay
        else:
            env_type = "ASR_sustained"  # Attack-Sustain-Release with sustained volume
    
    return (A, S, R), env_type

def normalize_envelope(envelope, target_length=100):
    """
    Normalize an envelope to a standard length for comparison.
    
    Args:
        envelope: Input amplitude envelope
        target_length: Desired length of normalized envelope
        
    Returns:
        Normalized envelope of target_length
    """
    if len(envelope) <= 1:
        return np.zeros(target_length)
    
    # Normalize amplitude
    if np.max(envelope) > 0:
        envelope = envelope / np.max(envelope)
    
    # Normalize length using linear interpolation
    x_original = np.linspace(0, 1, len(envelope))
    x_target = np.linspace(0, 1, target_length)
    normalized = np.interp(x_target, x_original, envelope)
    
    return normalized