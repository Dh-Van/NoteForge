# main.py

import os
import wave
import numpy as np
import argparse
import matplotlib.pyplot as plt
from bss import separate
from test import main as classify_audio

def get_music(fpath: str, fname: str, max_seconds: float = 10.0) -> str:
    """
    Read up to the first `max_seconds` of stereo WAV at fpath/fname,
    convert to mono, and save to output/<song>/song_mono.wav.
    
    Args:
        fpath: Directory containing the audio file
        fname: Filename of the audio file
        max_seconds: Maximum length of audio to process (in seconds)
        
    Returns:
        Path to the saved mono WAV file
    """
    # Create full path to input file
    full_path = os.path.join(fpath, fname)
    
    print(f"[main] Processing {full_path}...")
    
    # Create output directory
    base_name = os.path.splitext(fname)[0]
    out_dir = os.path.join("output", base_name)
    os.makedirs(out_dir, exist_ok=True)
    
    # Path for mono output
    mono_path = os.path.join(out_dir, f"{base_name}_mono.wav")
    
    # Read the input WAV file
    try:
        inp = wave.open(full_path, 'rb')
        fs = inp.getframerate()
        total_f = inp.getnframes()
        nch = inp.getnchannels()
        sampw = inp.getsampwidth()
        
        # Calculate frames to read based on max_seconds
        frames_to_read = min(total_f, int(fs * max_seconds))
        
        # Read audio data
        raw = inp.readframes(frames_to_read)
        inp.close()
        
        # Convert to numpy array
        dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sampw]
        data = np.frombuffer(raw, dtype=dtype)
        
        # Convert to mono if stereo
        if nch == 2:
            data = data.reshape(-1, nch)
            mono = ((data[:, 0] + data[:, 1]) / 2.0).astype(dtype)
        else:
            mono = data
        
        # Write mono WAV file
        out = wave.open(mono_path, 'wb')
        out.setnchannels(1)
        out.setsampwidth(sampw)
        out.setframerate(fs)
        out.writeframes(mono.tobytes())
        out.close()
        
        # Print duration info
        duration = frames_to_read / fs
        print(f"[main] Processed {duration:.2f}s of audio (from {total_f/fs:.2f}s total)")
        print(f"[main] Mono audio saved to {mono_path}")
        
        return mono_path
        
    except Exception as e:
        print(f"[main] Error processing audio file: {e}")
        return None

def process_audio_file(audio_path, max_seconds=10.0, skip_separation=False, skip_classification=False):
    """
    Process an audio file through the entire pipeline:
    1. Convert to mono
    2. Perform blind source separation (BSS)
    3. Classify instruments
    
    Args:
        audio_path: Path to the audio file
        max_seconds: Maximum length of audio to process
        skip_separation: If True, skip BSS step (use existing audio.pkl)
        skip_classification: If True, skip classification step
        
    Returns:
        Dictionary containing output paths
    """
    results = {}
    
    # Split path into directory and filename
    fpath, fname = os.path.split(audio_path)
    if not fpath:
        fpath = "."
    
    # Step 1: Convert to mono
    mono_path = get_music(fpath, fname, max_seconds)
    if not mono_path:
        return None
    
    results['mono_path'] = mono_path
    
    # Step 2: Perform BSS
    if not skip_separation:
        print("[main] Running blind source separation...")
        audio_pkl_path = separate(mono_path)
        results['audio_pkl_path'] = audio_pkl_path
    else:
        # Look for existing audio.pkl in the output directory
        output_dir = os.path.dirname(mono_path)
        audio_pkl_path = os.path.join(output_dir, "audio.pkl")
        if os.path.exists(audio_pkl_path):
            print(f"[main] Using existing audio.pkl at {audio_pkl_path}")
            results['audio_pkl_path'] = audio_pkl_path
        else:
            print(f"[main] Error: Cannot skip separation step - {audio_pkl_path} not found")
            return results
    
    # Step 3: Classify instruments
    if not skip_classification:
        print("[main] Running instrument classification...")
        classification_results = classify_audio(audio_pkl_path)
        if classification_results:
            results.update(classification_results)
    
    print("[main] Processing complete!")
    return results

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Audio Decomposition: Music separation and classification')
    parser.add_argument('--input', '-i', type=str, help='Path to input WAV file', required=True)
    parser.add_argument('--max_seconds', '-t', type=float, default=10.0, help='Maximum seconds to process')
    parser.add_argument('--skip-separation', '-s', action='store_true', help='Skip blind source separation step')
    parser.add_argument('--skip-classification', '-c', action='store_true', help='Skip classification step')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process the audio file
    results = process_audio_file(
        args.input, 
        max_seconds=args.max_seconds,
        skip_separation=args.skip_separation,
        skip_classification=args.skip_classification
    )
    
    if results and 'piano_roll' in results:
        # Display the piano roll plot
        img = plt.imread(results['piano_roll'])
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()