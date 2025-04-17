from io import BytesIO
import requests
from pydub import AudioSegment
from zipfile import ZipFile
import os
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg
from scipy.ndimage import gaussian_filter1d
import utils
import regex as re

def get_all_instruments():
    data = [
        ("Violin_G", "https://theremin.music.uiowa.edu/sound%20files/MIS%20Pitches%20-%202014/Strings/Violin/Violin.arco.ff.sulG.stereo.zip"),
        ("Violin_D", "https://theremin.music.uiowa.edu/sound%20files/MIS%20Pitches%20-%202014/Strings/Violin/Violin.arco.ff.sulD.stereo.zip"),
        ("Violin_A", "https://theremin.music.uiowa.edu/sound%20files/MIS%20Pitches%20-%202014/Strings/Violin/Violin.arco.ff.sulA.stereo.zip"),
        ("Violin_E", "https://theremin.music.uiowa.edu/sound%20files/MIS%20Pitches%20-%202014/Strings/Violin/Violin.arco.ff.sulE.stereo.zip"),
        ("Viola_C", "https://theremin.music.uiowa.edu/sound%20files/MIS%20Pitches%20-%202014/Strings/Viola/Viola.arco.ff.sulC.stereo.zip"),
        ("Viola_G", "https://theremin.music.uiowa.edu/sound%20files/MIS%20Pitches%20-%202014/Strings/Viola/Viola.arco.ff.sulG.stereo.zip"),
        ("Viola_D", "https://theremin.music.uiowa.edu/sound%20files/MIS%20Pitches%20-%202014/Strings/Viola/Viola.arco.ff.sulD.stereo.zip"),
        ("Viola_A", "https://theremin.music.uiowa.edu/sound%20files/MIS%20Pitches%20-%202014/Strings/Viola/Viola.arco.ff.sulA.stereo.zip"),
        ("Cello_C", "https://theremin.music.uiowa.edu/sound%20files/MIS%20Pitches%20-%202014/Strings/Cello/Cello.arco.ff.sulC.stereo.zip"),        
        ("Cello_G", "https://theremin.music.uiowa.edu/sound%20files/MIS%20Pitches%20-%202014/Strings/Cello/Cello.arco.ff.sulG.stereo.zip"),        
        ("Cello_D", "https://theremin.music.uiowa.edu/sound%20files/MIS%20Pitches%20-%202014/Strings/Cello/Cello.arco.ff.sulD.stereo.zip"),        
        ("Cello_A", "https://theremin.music.uiowa.edu/sound%20files/MIS%20Pitches%20-%202014/Strings/Cello/Cello.arco.ff.sulA.stereo.zip"),        
        ("Bass_C", "https://theremin.music.uiowa.edu/sound%20files/MIS%20Pitches%20-%202014/Strings/Double%20Bass/Bass.arco.ff.sulC.stereo.zip"),
        ("Bass_E", "https://theremin.music.uiowa.edu/sound%20files/MIS%20Pitches%20-%202014/Strings/Double%20Bass/Bass.arco.ff.sulE.stereo.zip"),
        ("Bass_A", "https://theremin.music.uiowa.edu/sound%20files/MIS%20Pitches%20-%202014/Strings/Double%20Bass/Bass.arco.ff.sulA.stereo.zip"),
        ("Bass_D", "https://theremin.music.uiowa.edu/sound%20files/MIS%20Pitches%20-%202014/Strings/Double%20Bass/Bass.arco.ff.sulD.stereo.zip"),
        ("Bass_G", "https://theremin.music.uiowa.edu/sound%20files/MIS%20Pitches%20-%202014/Strings/Double%20Bass/Bass.arco.ff.sulG.stereo.zip"),
    ]

    for instrument, url in data:
        get_single_instrument_sample(instrument, url)

def get_single_instrument_sample(instrument, url):
    os.makedirs(f'instrument_samples/{instrument[:-2]}/{instrument}', exist_ok=True)
    response = requests.get(url)

    if(response.status_code != 200):
        print(f'ERROR | Status Code ${response.status_code}')
        return
    
    zip_data = BytesIO(response.content)
    with ZipFile(zip_data) as all_samples:
        for name in all_samples.namelist():
            if "__MACOSX" in name: continue
            print(f'Converting {name}')
            aiff_data = BytesIO(all_samples.read(name))
            audio = AudioSegment.from_file(aiff_data, format="aiff")
            name_arr = name.split(".")
            fname = f'{name_arr[0]}.{name_arr[3]}.{name_arr[4]}'
            audio.export(f'instrument_samples/{instrument[:-2]}/{instrument}/{fname}.wav', format="wav")

def compute_envelope(signal, sampling_rate, freq=440, chunk_size=20):
    # Bandpass Filter around the note's freq
    cutoff = (freq / np.sqrt(2), freq * np.sqrt(2))
    sos = sg.butter(5, cutoff, btype="bandpass", fs=sampling_rate, output='sos')
    filtered = sg.sosfilt(sos, signal)

    magnitude = np.abs(filtered)
    envelope = []
    for i in range(0, len(filtered), chunk_size):
        chunk = filtered[i : i + chunk_size]
        envelope.append(chunk)

    # # Step 4: Smooth envelope with Gaussian filter
    # smoothed = gaussian_filter1d(trimmed, sigma=5)

    return envelope

def retrive_asr(envelope):
    pass

def save_single_asr(instrument, sul, note, waveform, sampling_rate):
    split_note = re.match(r'^([A-G])(\d+)$', note, re.IGNORECASE)
    note_letter, octave = split_note.groups()

    freq = utils.note_to_freq(note_letter, octave)

    envelope = compute_envelope(waveform, sampling_rate, freq)

    # attack, sustain, release = retrive_asr(envelope)

    plt.figure()
    plt.plot(envelope)
    plt.show()

    # key = (instrument.lower(), note.upper())
    # if key not in refrence_db:
    #     refrence_db[key] = {}

    # refrence_db[key][sul] = {
    #     "waveform": waveform,
    #     "sampling_rate": sampling_rate,
    #     "envelope": envelope,
    #     "attack": attack,
    #     "sustain": sustain,
    #     "release": release
    # }


refrence_db = {}

waveform, sampling_rate = sf.read("./instrument_samples/Viola/Viola_A/Viola.sulA.A4.wav")

save_single_asr("Viola", "A", "A4", waveform, sampling_rate)
