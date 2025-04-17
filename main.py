import wave
import numpy as np
import matplotlib.pyplot as plt
import os
import bss

'''
params: Takes in a file path to a .wav music file
returns: The mono audio data for that music file
'''
def get_music(fpath, fname):
    music = wave.open(f'{fpath}/{fname}', 'rb')

    num_frames = music.getnframes()

    raw_data = music.readframes(num_frames)
    music.close()

    audio_data = np.frombuffer(raw_data, dtype=np.int16)

    left_channel = audio_data[0::2].astype(np.float32)
    right_channel = audio_data[1::2].astype(np.float32)
    mono_float = (left_channel + right_channel) / 2.0
    mono = np.clip(mono_float, -32768, 32767).astype(np.int16)

    os.makedirs(f'./output/{fname[:-4]}', exist_ok=True)

    output_wave = wave.open(f'./output/{fname[:-4]}/{fname[:-4]}_mono.wav', 'wb')

    output_wave.setnchannels(1)
    output_wave.setsampwidth(music.getsampwidth())
    output_wave.setframerate(music.getframerate())

    output_wave.writeframes(mono.tobytes())
    output_wave.close()

    return f'./output/{fname[:-4]}/{fname[:-4]}_mono.wav'

def separate_audio(fpath, fname):
    out_path = get_music(fpath, fname)
    print(out_path)
    bss.separate(out_path)
    return True

def convert_to_notes(fpath):
    pass

def gen_sheet_music(fpath, fname):
    pass    

def cli():
    # Input phrases
    option_phrase = f'Select an input (1-3): \n1: Separate Audio\n2: Convert to Notes\n3: Generate Sheet Music\nx: Exit Program\n'
    file_phrase = f'Enter file path to .wav file:\n'
    separated_phrase = f'Enter path to saperated audio files folder:\n'
    option = input(option_phrase)
    if(option == "1"):
        file_input = input(file_phrase)
        fpath, fname = os.path.split(file_input)
        success = separate_audio(fpath, fname)
        if(success):
            print("Audio separated, exiting program")
            cli()
        else:
            print("Separation error, exiting program")
            cli()
    elif(option == "2"):
        separated_input = input(separated_phrase)
        convert_to_notes(separated_input)
        pass
    elif(option == "3"):
        pass
    elif(option == "X" or option == "x"):
        return
    else:
        cli()

# seperate_audio("./input", "SwampThang.wav")
cli()

