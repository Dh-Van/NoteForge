# main.py

import wave
import numpy as np
import os
import bss

"""
params: Takes in a file path to a .wav music file
returns: The mono audio data for that music file
"""
def get_music(fpath, fname):
    music = wave.open(f'{fpath}/{fname}', 'rb')
    num_frames = music.getnframes()
    raw_data = music.readframes(num_frames)
    music.close()

    audio_data   = np.frombuffer(raw_data, dtype=np.int16)
    left_channel = audio_data[0::2].astype(np.float32)
    right_channel= audio_data[1::2].astype(np.float32)
    mono_float   = (left_channel + right_channel) / 2.0
    mono         = np.clip(mono_float, -32768, 32767).astype(np.int16)

    out_dir = f'./output/{fname[:-4]}'
    os.makedirs(out_dir, exist_ok=True)

    output_wave = wave.open(f'{out_dir}/{fname[:-4]}_mono.wav', 'wb')
    output_wave.setnchannels(1)
    output_wave.setsampwidth(music.getsampwidth())
    output_wave.setframerate(music.getframerate())
    output_wave.writeframes(mono.tobytes())
    output_wave.close()

    return f'{out_dir}/{fname[:-4]}_mono.wav'


def separate_audio(fpath, fname):
    mono_path = get_music(fpath, fname)
    print(f"Mono audio written to {mono_path}")
    bss.separate(mono_path)
    print("STFT magnitude matrix and audio.pkl generated.")
    return True


def convert_to_notes(folder_path):
    # TODO: implement conversion from separated audio to MIDI/notes
    pass


def gen_sheet_music(folder_path, fname):
    # TODO: implement sheet music generation from note data
    pass


def cli():
    option_phrase    = (
        "Select an input (1-3):\n"
        "  1: Separate Audio\n"
        "  2: Convert to Notes\n"
        "  3: Generate Sheet Music\n"
        "  x: Exit Program\n"
    )
    file_phrase      = "Enter file path to .wav file:\n"
    separated_phrase = "Enter path to separated audio files folder:\n"

    option = input(option_phrase)
    if option == "1":
        file_input   = input(file_phrase)
        fpath, fname = os.path.split(file_input)
        success      = separate_audio(fpath, fname)
        print("Audio separated." if success else "Separation error.")
        cli()

    elif option == "2":
        folder = input(separated_phrase)
        convert_to_notes(folder)
        cli()

    elif option == "3":
        folder, fname = input(separated_phrase).rsplit(os.sep, 1)
        gen_sheet_music(folder, fname)
        cli()

    elif option.lower() == "x":
        return

    else:
        cli()


if __name__ == "__main__":
    # quick test run:
    separate_audio("./input", "SugarPlum.wav")
    # to enable interactive CLI, uncomment:
    # cli()
