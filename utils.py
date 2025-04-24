import re
import numpy as np

# Mapping from note names to semitone offsets within an octave
_SEMITONE_MAP = {
    "C": 0,  "Db": 1, "D": 2,  "Eb": 3,
    "E": 4,  "F": 5,  "Gb": 6, "G": 7,
    "Ab": 8, "A": 9,  "Bb": 10,"B": 11
}

# Instrument name lists for sample classification
STRINGS   = ["Violin", "Viola", "Cello", "Bass"]
WOODWINDS = ["Flute",]
CLARINET  = ["BbClarinet"]


def note_to_freq(note: str, octave: str) -> float:
    """
    Convert a musical note (e.g., "C", "Db", "A") and octave (e.g., "4")
    to its frequency in Hz (A4 = 440 Hz).
    """
    n = note.strip()
    try:
        o = int(octave)
    except ValueError:
        raise ValueError(f"Invalid octave: {octave}")
    if n not in _SEMITONE_MAP:
        raise ValueError(f"Unknown note name: {n}")

    semitone = _SEMITONE_MAP[n]
    midi_number = 12 * (o + 1) + semitone
    return 440.0 * 2 ** ((midi_number - 69) / 12)


def split_note_name(full_note: str) -> tuple[str, int]:
    """
    Split a string like 'Bb4' or 'C#5' into (note, octave).
    Returns (note_str, octave_int).
    """
    m = re.match(r'^([A-G][b#]?)(\d+)$', full_note)
    if not m:
        raise ValueError(f"Cannot parse note: {full_note}")
    note_part, octave_part = m.groups()
    return note_part, int(octave_part)
