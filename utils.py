def note_to_freq(note, octave):
    tone_dict = {
        "C": 0,
        "Db": 1,
        "D": 2,
        "Eb": 3,
        "E": 4,
        "F": 5,
        "Gb": 6,
        "G": 7,
        "Ab": 8,
        "A": 9,
        "Bb": 10,
        "B": 11,
    }

    midi_number = 12 * (int(octave) + 1) + tone_dict[note]

    freq = 440.0 * 2 ** ((midi_number - 69) / 12)
    return freq

def generate_note_list():
    tones = ["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"]

    all_notes = [f"{t}{octave}" for octave in range(1, 7) for t in tones]

    start = all_notes.index("A1")
    end   = all_notes.index("G6") + 1
    return all_notes[start:end]

STRINGS = [
    "Violin",
    "Viola",
    "Cello",
    "Bass"
]

WOODWINDS = [
    "Flute",
]

CLARINET = [
    "BbClarinet",
]