import pickle
import matplotlib.pyplot as plt
import os
import soundfile as sf
import utils


reference_db = {}
with open('reference_db.pkl', 'rb') as f:
    reference_db = pickle.load(f)

notes = []
with open('audio.pkl', 'rb') as f:
    notes = pickle.load(f)


all_instruments = reference_db.keys()

for i, note in enumerate(utils.generate_note_list()):
    audio = notes[i]
    for instrument_key in all_instruments:
        if(instrument_key[1] != note): 
            continue
        instrument = reference_db[instrument_key]
        sul_key = instrument.keys()
        for sul in sul_key:
            instrument_data = instrument[sul]
            print(instrument_data["attack"])