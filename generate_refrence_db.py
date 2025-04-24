from io import BytesIO
import requests
from pydub import AudioSegment
from zipfile import ZipFile
import os
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg
import utils
import regex as re
import pickle

def get_all_instruments():
    data = [
        # ("Violin_G", "https://theremin.music.uiowa.edu/sound%20files/MIS%20Pitches%20-%202014/Strings/Violin/Violin.arco.ff.sulG.stereo.zip"),
        # ("Violin_D", "https://theremin.music.uiowa.edu/sound%20files/MIS%20Pitches%20-%202014/Strings/Violin/Violin.arco.ff.sulD.stereo.zip"),
        # ("Violin_A", "https://theremin.music.uiowa.edu/sound%20files/MIS%20Pitches%20-%202014/Strings/Violin/Violin.arco.ff.sulA.stereo.zip"),
        # ("Violin_E", "https://theremin.music.uiowa.edu/sound%20files/MIS%20Pitches%20-%202014/Strings/Violin/Violin.arco.ff.sulE.stereo.zip"),
        # ("Viola_C", "https://theremin.music.uiowa.edu/sound%20files/MIS%20Pitches%20-%202014/Strings/Viola/Viola.arco.ff.sulC.stereo.zip"),
        # ("Viola_G", "https://theremin.music.uiowa.edu/sound%20files/MIS%20Pitches%20-%202014/Strings/Viola/Viola.arco.ff.sulG.stereo.zip"),
        # ("Viola_D", "https://theremin.music.uiowa.edu/sound%20files/MIS%20Pitches%20-%202014/Strings/Viola/Viola.arco.ff.sulD.stereo.zip"),
        # ("Viola_A", "https://theremin.music.uiowa.edu/sound%20files/MIS%20Pitches%20-%202014/Strings/Viola/Viola.arco.ff.sulA.stereo.zip"),
        # ("Cello_C", "https://theremin.music.uiowa.edu/sound%20files/MIS%20Pitches%20-%202014/Strings/Cello/Cello.arco.ff.sulC.stereo.zip"),        
        # ("Cello_G", "https://theremin.music.uiowa.edu/sound%20files/MIS%20Pitches%20-%202014/Strings/Cello/Cello.arco.ff.sulG.stereo.zip"),        
        # ("Cello_D", "https://theremin.music.uiowa.edu/sound%20files/MIS%20Pitches%20-%202014/Strings/Cello/Cello.arco.ff.sulD.stereo.zip"),        
        # ("Cello_A", "https://theremin.music.uiowa.edu/sound%20files/MIS%20Pitches%20-%202014/Strings/Cello/Cello.arco.ff.sulA.stereo.zip"),        
        # ("Bass_C", "https://theremin.music.uiowa.edu/sound%20files/MIS%20Pitches%20-%202014/Strings/Double%20Bass/Bass.arco.ff.sulC.stereo.zip"),
        # ("Bass_E", "https://theremin.music.uiowa.edu/sound%20files/MIS%20Pitches%20-%202014/Strings/Double%20Bass/Bass.arco.ff.sulE.stereo.zip"),
        # ("Bass_A", "https://theremin.music.uiowa.edu/sound%20files/MIS%20Pitches%20-%202014/Strings/Double%20Bass/Bass.arco.ff.sulA.stereo.zip"),
        # ("Bass_D", "https://theremin.music.uiowa.edu/sound%20files/MIS%20Pitches%20-%202014/Strings/Double%20Bass/Bass.arco.ff.sulD.stereo.zip"),
        # ("Bass_G", "https://theremin.music.uiowa.edu/sound%20files/MIS%20Pitches%20-%202014/Strings/Double%20Bass/Bass.arco.ff.sulG.stereo.zip"),
        ("Flute_V", "https://theremin.music.uiowa.edu/sound%20files/MIS%20Pitches%20-%202014/Woodwinds/Flute/Flute.vib.ff.stereo.zip"),
        ("Flute_N", "https://theremin.music.uiowa.edu/sound%20files/MIS%20Pitches%20-%202014/Woodwinds/Flute/Flute.nonvib.ff.stereo.zip"),
        ("Clarinet_b", "https://theremin.music.uiowa.edu/sound%20files/MIS%20Pitches%20-%202014/Woodwinds/Bb%20Clarinet/BbClarinet.ff.stereo.zip")
    ]

    # data = data[

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
            if(any(i in name_arr for i in utils.STRINGS)):
                fname = f'{name_arr[0]}.{name_arr[3]}.{name_arr[4]}'
            if(any(i in name_arr[0] for i in utils.WOODWINDS)):
                fname = f'{name_arr[0]}.{name_arr[1]}.{name_arr[3]}'
            if(any(i in name_arr for i in utils.CLARINET)):
                fname = f'{name_arr[0]}.{name_arr[1]}.{name_arr[2]}'
            audio.export(f'instrument_samples/{instrument[:-2]}/{instrument}/{fname}.wav', format="wav")

# Envelope extraction parameters
SecondDerivThreshold = 0.02
dilation = (3, 5)

def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    """
    Returns local minima and maxima indices for envelope seeding.
    """
    lmin = (np.diff(np.sign(np.diff(s))) >= 0).nonzero()[0] + 1
    lmax = (np.diff(np.sign(np.diff(s))) <= 0).nonzero()[0] + 1
    if split:
        mid = np.mean(s)
        lmin = lmin[s[lmin] < mid]
        lmax = lmax[s[lmax] > mid]
    lmin = lmin[[i + np.argmin(s[lmin[i:i + dmin]])
                 for i in range(0, len(lmin), dmin)]]
    lmax = lmax[[i + np.argmax(s[lmax[i:i + dmax]])
                 for i in range(0, len(lmax), dmax)]]
    return lmin, lmax

def compute_envelope(data, samplerate, cutoff,
                     n_chunks=500, pad_chunks=1, max_iter=200):
    """
    Compute the amplitude envelope and ASR boundaries.

    Parameters:
        data        : 1D numpy array of audio samples
        samplerate  : sample rate in Hz
        cutoff      : (low, high) tuple in Hz for bandpass
        n_chunks    : target number of envelope chunks
        pad_chunks  : padding size (in chunks) for seeding
        max_iter    : max iterations for upper‐envelope fitting

    Returns:
        indices  : np.array([A, B, C, D])  # sample indices for ASR
        types    : [“AS”/“ASR”, “Static”/“Dynamic”]
        env_curve: 1D numpy array of full‑length envelope
    """
    low, high = cutoff

    # 1) Bandpass filter & rectify
    sos      = sg.butter(5, (low, high), fs=samplerate,
                         btype='bandpass', output='sos')
    filtered = sg.sosfilt(sos, data)
    rect     = np.abs(filtered)

    # 2) Chunk‐wise max downsampling (avoid empty slices)
    chunk_size = int(np.ceil(len(rect) / n_chunks))
    n_actual   = int(np.ceil(len(rect) / chunk_size))
    rect_chunks = np.array([
        rect[i*chunk_size : min((i+1)*chunk_size, len(rect))].max()
        for i in range(n_actual)
    ])

    # 3) Pad for envelope seeding
    padded = np.pad(rect_chunks, pad_chunks, constant_values=0.0)
    M      = len(padded)
    x      = np.arange(M)

    # 4) Seed points: local extrema + first/last above threshold
    _, extrema = hl_envelopes_idx(padded, dmin=1, dmax=1)
    A_c, D_c   = np.where(padded > 0.01 * padded.max())[0][[0, -1]]
    seeds      = np.sort(np.unique(np.concatenate([extrema, [A_c, D_c]])))

    # 5) Iteratively fit upper envelope on chunk‐level data
    prev_len = -1
    for _ in range(max_iter):
        if len(seeds) == prev_len:
            break
        prev_len = len(seeds)
        interp   = np.interp(x, seeds, padded[seeds])
        diff     = padded - interp
        idx      = np.argmax(diff)
        if padded[idx] <= interp[idx]:
            break
        seeds    = np.sort(np.append(seeds, idx))
    absinterp = interp

    # 6) Second derivatives for ASR boundary detection
    fp    = np.diff(absinterp)
    fp_sm = np.convolve(fp, np.ones(dilation[0]), mode='same')
    fpp   = np.diff(fp_sm)
    fpp_sm = np.convolve(fpp, np.ones(dilation[1]), mode='same')

    # 7) Find chunk‐indices B_c and C_c within [A_c, D_c]
    segment = fpp_sm[A_c : D_c+1]
    neg     = (-segment).clip(min=0)
    peaks   = np.where(neg > (neg.max() / 20))[0]
    B_c     = A_c + (peaks[0] if len(peaks) else 0)

    peaks2  = np.where(neg > 0)[0]
    C_c     = A_c + (peaks2[-1] if len(peaks2) else (D_c - A_c))

    # 8) Classify AS vs ASR and Static vs Dynamic
    types = [
        'AS'  if C_c <= B_c else 'ASR',
        'Dynamic' if (absinterp[C_c] - absinterp[B_c]) < 0 else 'Static'
    ]

    # 9) Map chunk indices back to sample indices
    A = max((A_c - pad_chunks) * chunk_size, 0)
    B = max((B_c - pad_chunks) * chunk_size, 0)
    C = max((C_c - pad_chunks) * chunk_size, 0)
    D = min((D_c - pad_chunks) * chunk_size, len(data) - 1)
    indices = np.array([A, B, C, D])

    # 10) Upsample envelope back to per‑sample resolution
    env_chunks     = absinterp[pad_chunks : M - pad_chunks]
    chunk_positions = np.linspace(0, len(data)-1, num=env_chunks.size)
    env_curve      = np.interp(np.arange(len(data)), chunk_positions, env_chunks)

    return indices, types, env_curve


def save_single_asr(db, instrument, sul, note, waveform, sampling_rate):
    split_note = re.match(r'^([A-Za-z]+)(\d+)$', note)
    note_letter, octave = split_note.groups()

    freq = utils.note_to_freq(note_letter, octave)

    idx, types, envelope = compute_envelope(waveform, sampling_rate, (freq / np.sqrt(2), freq * np.sqrt(2)))

    key = (instrument.lower(), note.upper())
    if key not in db:
        db[key] = {}

    db[key][sul] = {
        "waveform": waveform,
        "sampling_rate": sampling_rate,
        "envelope": envelope,
        "attack": envelope[idx[0]:idx[1]],
        "sustain": envelope[idx[1]:idx[2]],
        "release": envelope[idx[2]:idx[3]]
    }


def save_all_asr(reference_db):
    root = "instrument_samples"
    for instrument_folder in os.listdir(root):
        for sul_folder in os.listdir(os.path.join(root, instrument_folder)):
            for note_file in os.listdir(os.path.join(root, instrument_folder, sul_folder)):
                sul = sul_folder[-1]
                instrument = instrument_folder
                note = note_file.split(".")[-2]
                waveform, sampling_rate = sf.read(os.path.join(root, instrument_folder, sul_folder, note_file))
                save_single_asr(reference_db, instrument, sul, note, waveform, sampling_rate)

def plot_from_db(db, instrument, sul, note):
    info = db[(instrument, note)][sul]
    print(info)
    plt.figure()
    plt.subplot(4, 1, 1)
    plt.plot(info["envelope"])
    plt.subplot(4, 1, 2)
    plt.plot(info["attack"])
    plt.subplot(4, 1, 3)
    plt.plot(info["sustain"])
    plt.subplot(4, 1, 4)
    plt.plot(info["release"])
    plt.show()

def create_db():
    reference_db = {}
    save_all_asr(reference_db)

    with open("reference_db.pkl", "wb") as f:
        pickle.dump(reference_db, f, protocol=pickle.HIGHEST_PROTOCOL)

create_db()
# get_all_instruments()