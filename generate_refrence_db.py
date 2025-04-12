from io import BytesIO
import requests
from pydub import AudioSegment
from zipfile import ZipFile
import os
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

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
    os.makedirs(f'instrument_samples/{instrument}', exist_ok=True)
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
            audio.export(f'instrument_samples/{instrument}/{fname}.wav', format="wav")


def compute_envelope(signal, chunk_size=256, threshold=1e-2):
    envelope = []
    for i in range(0, len(signal), chunk_size):
        chunk = signal[i : i + chunk_size]
        envelope.append(np.max(chunk))
    zero_idx = np.where(np.array(envelope) > threshold)[0]
    return gaussian_filter1d(envelope[zero_idx[0] : zero_idx[-1] + 1], 2)

def retrive_asr(envelope):
    env = envelope / (np.max(envelope))
    env_length = len(env)
    slope = np.gradient(env)
    curvature = np.gradient(slope)

    attack_end_index = 0.2 * env_length
    release_start_index = 0.8 * env_length

    for i in range(1, env_length):
        if(slope[i] < slope[i - 1] and curvature[i] < 0):
            attack_end_index = i
            break

    for i in range(env_length - 2, 0, -1):
        if(slope[i] > slope[i + 1] or curvature[i] > 0):
            release_start_index = i
            break

    return env[:attack_end_index], env[attack_end_index:release_start_index], env[release_start_index:]

waveform, sampling_rate = sf.read("./instrument_samples/Viola_A/Viola.sulA.A4.wav")
a, s, r = retrive_asr(compute_envelope(waveform))

plt.figure()
plt.subplot(4, 1, 1)
plt.plot(np.concatenate([a, s, r]))
plt.subplot(4, 1, 2)
plt.plot(a)
plt.subplot(4, 1, 3)
plt.plot(s)
plt.subplot(4, 1, 4)
plt.plot(r)

waveform, sampling_rate = sf.read("./instrument_samples/Bass_C/Bass.sulC.C1.wav")
a, s, r = retrive_asr(compute_envelope(waveform))

plt.figure()
plt.subplot(4, 1, 1)
plt.plot(np.concatenate([a, s, r]))
plt.subplot(4, 1, 2)
plt.plot(a)
plt.subplot(4, 1, 3)
plt.plot(s)
plt.subplot(4, 1, 4)
plt.plot(r)
plt.show()
    