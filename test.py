# test_fixed_all_notes.py
# Modified version that shows all notes on the axis

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter
from utils import note_to_midi

# ─── Classification Parameters ─────────────────────────────
SILENCE_THRESHOLD = 0.01      # Ignore magnitudes below this value
EVENT_MIN_DURATION = 20        # Minimum event length in frames

# Instrument playable ranges (in MIDI note numbers)
# Flute range: B3 (59) to D7 (98)
FLUTE_RANGE = (59, 98)  
# Clarinet range: D3 (50) to G6 (91)
CLARINET_RANGE = (50, 91)
# ──────────────────────────────────────────────────────────

def load_reference_db(path='reference_db.pkl'):
    """Load the reference database containing instrument envelope templates"""
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except:
        print("[test] Could not load reference database, will use forced classification")
        return {}

def load_audio_data(path='audio.pkl'):
    """Load the processed audio data"""
    with open(path, 'rb') as f:
        return pickle.load(f)

def segment_events(mag, threshold=SILENCE_THRESHOLD):
    """
    Find contiguous segments in magnitude data above threshold.
    
    Args:
        mag: Magnitude data for a single note
        threshold: Silence threshold
        
    Returns:
        List of (start, end) frame indices
    """
    events = []
    in_event = False
    start = 0
    
    for i, val in enumerate(mag):
        if not in_event and val > threshold:
            # Start of new event
            in_event = True
            start = i
        elif in_event and val <= threshold:
            # End of current event
            if i - start >= EVENT_MIN_DURATION:
                events.append((start, i))
            in_event = False
    
    # Handle case where audio ends during an event
    if in_event and len(mag) - start >= EVENT_MIN_DURATION:
        events.append((start, len(mag)))
    
    return events

def note_in_range(note, range_tuple):
    """Check if a note is within a given MIDI range"""
    midi_num = note_to_midi(note)
    return range_tuple[0] <= midi_num <= range_tuple[1]

def is_in_flute_range(note):
    """Check if a note is in flute range"""
    return note_in_range(note, FLUTE_RANGE)

def is_in_clarinet_range(note):
    """Check if a note is in clarinet range"""
    return note_in_range(note, CLARINET_RANGE)

def classify_events_with_ranges(data, ref_db=None):
    """
    Classify notes based on playable ranges and enforce one note per time frame per instrument.
    
    Args:
        data: Audio data dictionary from audio.pkl
        ref_db: Not used in this version
        
    Returns:
        Classification map
    """
    notes = data['notes']
    mag_matrix = data['mag_matrix']
    times = data['times']
    
    n_frames, n_notes = mag_matrix.shape
    
    # Initialize classification map (0=silence, 1=flute, 2=clarinet)
    class_map = np.zeros((n_frames, n_notes), dtype=int)
    
    # Convert notes to MIDI numbers for range checking
    midi_numbers = [note_to_midi(note) for note in notes]
    
    # Track statistics for debugging
    stats = {
        "total_events": 0, 
        "classified_events": 0, 
        "instruments": {"flute": 0, "clarinet": 0},
        "out_of_range": 0
    }
    
    # First, find all events
    all_events = []
    for j, note in enumerate(notes):
        # Skip if no significant energy in this note
        if np.max(mag_matrix[:, j]) < SILENCE_THRESHOLD:
            continue
        
        # Get magnitude data for this note
        mag = mag_matrix[:, j]
        
        # Find segments with significant energy
        events = segment_events(mag)
        
        # Skip if no events found
        if not events:
            continue
        
        # Store events with note information and magnitude
        for start, end in events:
            avg_mag = np.mean(mag[start:end])
            all_events.append({
                'note_idx': j,
                'note': note,
                'midi': midi_numbers[j],
                'start': start,
                'end': end,
                'magnitude': avg_mag,
                'in_flute_range': is_in_flute_range(note),
                'in_clarinet_range': is_in_clarinet_range(note)
            })
    
    # Count total events
    stats["total_events"] = len(all_events)
    print(f"Found {len(all_events)} total events")
    
    # Sort events by start time for processing
    all_events.sort(key=lambda e: e['start'])
    
    # Process events one time frame at a time
    for t in range(n_frames):
        # Get events active at this time frame
        active_events = [e for e in all_events if e['start'] <= t < e['end']]
        
        if not active_events:
            continue
            
        # Sort by magnitude (loudest first)
        active_events.sort(key=lambda e: e['magnitude'], reverse=True)
        
        # Variables to track if we've assigned a note to each instrument in this frame
        flute_assigned = False
        clarinet_assigned = False
        
        # Try to assign notes to instruments
        for event in active_events:
            note_idx = event['note_idx']
            
            # Skip if this note is already classified in this frame
            if class_map[t, note_idx] != 0:
                continue
                
            # Try to assign to flute first if in range and flute not yet assigned
            if not flute_assigned and event['in_flute_range']:
                class_map[t, note_idx] = 1  # Flute
                flute_assigned = True
                if t == event['start']:  # Count only once per event
                    stats["instruments"]["flute"] += 1
                    stats["classified_events"] += 1
                    
            # Otherwise try clarinet if in range and clarinet not yet assigned
            elif not clarinet_assigned and event['in_clarinet_range']:
                class_map[t, note_idx] = 2  # Clarinet
                clarinet_assigned = True
                if t == event['start']:  # Count only once per event
                    stats["instruments"]["clarinet"] += 1
                    stats["classified_events"] += 1
            
            # If event is out of both ranges, count it but don't classify
            elif t == event['start'] and not event['in_flute_range'] and not event['in_clarinet_range']:
                stats["out_of_range"] += 1
    
    # Print classification statistics
    print(f"[test] Classification complete:")
    print(f"  Total events: {stats['total_events']}")
    print(f"  Classified events: {stats['classified_events']} ({100*stats['classified_events']/max(1, stats['total_events']):.1f}%)")
    print(f"  Out of range events: {stats['out_of_range']} ({100*stats['out_of_range']/max(1, stats['total_events']):.1f}%)")
    for inst, count in stats["instruments"].items():
        print(f"  {inst.capitalize()}: {count} events ({100*count/max(1, stats['classified_events']):.1f}%)")
    
    return class_map

def plot_classification(notes, times, class_map, output_path=None):
    """
    Plot the classification results.
    
    Args:
        notes: List of note names
        times: Array of time points
        class_map: Classification map (0=silence, 1=flute, 2=clarinet)
        output_path: Path to save the plot (if None, display instead)
        
    Returns:
        Path to saved plot if output_path is provided
    """
    n_frames, n_notes = class_map.shape
    
    # Apply Gaussian smoothing for better visualization
    smoothed_map = gaussian_filter(class_map.astype(float), sigma=(1.0, 0))
    
    # Create color map
    cmap = mcolors.ListedColormap(['white', 'tab:blue', 'tab:green'])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    # Create figure with adjusted size for better visibility
    plt.figure(figsize=(14, 10))
    
    # Plot the classification map
    plt.imshow(
        smoothed_map.T,  # Transpose for better orientation
        aspect='auto',
        origin='lower',
        extent=[times[0], times[-1], 0, n_notes],
        cmap=cmap,
        norm=norm,
        interpolation='nearest'
    )
    
    # Add range indicators for instruments
    midi_values = [note_to_midi(note) for note in notes]
    
    # Find indices corresponding to range boundaries
    flute_low_idx = next((i for i, m in enumerate(midi_values) if m >= FLUTE_RANGE[0]), None)
    flute_high_idx = next((i for i, m in enumerate(midi_values) if m > FLUTE_RANGE[1]), len(midi_values))
    clarinet_low_idx = next((i for i, m in enumerate(midi_values) if m >= CLARINET_RANGE[0]), None)
    clarinet_high_idx = next((i for i, m in enumerate(midi_values) if m > CLARINET_RANGE[1]), len(midi_values))
    
    # Add range indicators
    if flute_low_idx is not None:
        plt.axhline(y=flute_low_idx, color='tab:blue', linestyle='--', alpha=0.5)
        plt.axhline(y=flute_high_idx, color='tab:blue', linestyle='--', alpha=0.5)
    if clarinet_low_idx is not None:
        plt.axhline(y=clarinet_low_idx, color='tab:green', linestyle='--', alpha=0.5)
        plt.axhline(y=clarinet_high_idx, color='tab:green', linestyle='--', alpha=0.5)
    
    # Add labels and title
    plt.xlabel('Time (s)')
    plt.ylabel('Note')
    plt.title('Instrument Classification (with Playable Ranges)')
    
    # Add y-tick labels (note names) - SHOW ALL NOTES
    plt.yticks(
        np.arange(n_notes) + 0.5,
        notes
    )
    
    # Add legend
    legend_patches = [
        mpatches.Patch(color='white', label='Silence'),
        mpatches.Patch(color='tab:blue', label='Flute'),
        mpatches.Patch(color='tab:green', label='Clarinet'),
        plt.Line2D([0], [0], color='tab:blue', linestyle='--', alpha=0.5, label='Flute Range'),
        plt.Line2D([0], [0], color='tab:green', linestyle='--', alpha=0.5, label='Clarinet Range')
    ]
    plt.legend(handles=legend_patches, loc='upper right')
    
    # Save or display
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        return output_path
    else:
        plt.tight_layout()
        plt.show()
        return None

def plot_magnitude(notes, times, mag_matrix, output_path=None):
    """
    Plot the magnitude data.
    
    Args:
        notes: List of note names
        times: Array of time points
        mag_matrix: Magnitude matrix
        output_path: Path to save the plot (if None, display instead)
        
    Returns:
        Path to saved plot if output_path is provided
    """
    n_frames, n_notes = mag_matrix.shape
    
    # Apply log scaling and normalization for better visualization
    mag_log = np.log1p(mag_matrix)
    if np.max(mag_log) > 0:
        mag_normalized = mag_log / np.max(mag_log)
    else:
        mag_normalized = mag_log
    
    # Apply Gaussian smoothing
    mag_smoothed = gaussian_filter(mag_normalized, sigma=(1.0, 0))
    
    # Create figure with adjusted size for better visibility
    plt.figure(figsize=(14, 10))
    
    # Plot the magnitude data
    plt.imshow(
        mag_smoothed.T,  # Transpose for better orientation
        aspect='auto',
        origin='lower',
        extent=[times[0], times[-1], 0, n_notes],
        cmap='viridis',
        interpolation='nearest'
    )
    
    # Add instrument range indicators
    midi_values = [note_to_midi(note) for note in notes]
    
    # Find indices corresponding to range boundaries
    flute_low_idx = next((i for i, m in enumerate(midi_values) if m >= FLUTE_RANGE[0]), None)
    flute_high_idx = next((i for i, m in enumerate(midi_values) if m > FLUTE_RANGE[1]), len(midi_values))
    clarinet_low_idx = next((i for i, m in enumerate(midi_values) if m >= CLARINET_RANGE[0]), None)
    clarinet_high_idx = next((i for i, m in enumerate(midi_values) if m > CLARINET_RANGE[1]), len(midi_values))
    
    # Add range indicators
    if flute_low_idx is not None:
        plt.axhline(y=flute_low_idx, color='tab:blue', linestyle='--', alpha=0.5)
        plt.axhline(y=flute_high_idx, color='tab:blue', linestyle='--', alpha=0.5)
    if clarinet_low_idx is not None:
        plt.axhline(y=clarinet_low_idx, color='tab:green', linestyle='--', alpha=0.5)
        plt.axhline(y=clarinet_high_idx, color='tab:green', linestyle='--', alpha=0.5)
    
    # Add labels and title
    plt.xlabel('Time (s)')
    plt.ylabel('Note')
    plt.title('Magnitude Spectrogram (with Instrument Ranges)')
    
    # Add y-tick labels (note names) - SHOW ALL NOTES
    plt.yticks(
        np.arange(n_notes) + 0.5,
        notes
    )
    
    # Add legend for instrument ranges
    legend_elements = [
        plt.Line2D([0], [0], color='tab:blue', linestyle='--', alpha=0.5, label='Flute Range'),
        plt.Line2D([0], [0], color='tab:green', linestyle='--', alpha=0.5, label='Clarinet Range')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Add colorbar
    plt.colorbar(label='Normalized Magnitude (log scale)')
    
    # Save or display
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        return output_path
    else:
        plt.tight_layout()
        plt.show()
        return None

def plot_piano_roll(notes, times, class_map, output_path=None):
    """
    Create a piano roll visualization of the classification results.
    
    Args:
        notes: List of note names
        times: Array of time points
        class_map: Classification map (0=silence, 1=flute, 2=clarinet)
        output_path: Path to save the plot (if None, display instead)
        
    Returns:
        Path to saved plot if output_path is provided
    """
    n_frames, n_notes = class_map.shape
    
    # Create figure with adjusted size for better visibility of all notes
    plt.figure(figsize=(14, 12))
    
    # Create color map for different instruments
    colors = ['tab:blue', 'tab:green']
    
    # Find non-zero (active) notes for each instrument
    for instrument_id in range(1, 3):  # 1=flute, 2=clarinet
        for j, note in enumerate(notes):
            i = 0
            while i < n_frames:
                if class_map[i, j] == instrument_id:
                    # Find end of this note
                    start_idx = i
                    end_idx = start_idx
                    while end_idx < n_frames - 1 and class_map[end_idx + 1, j] == instrument_id:
                        end_idx += 1
                    
                    # Plot this note segment
                    plt.plot(
                        [times[start_idx], times[end_idx + 1]], 
                        [j, j], 
                        color=colors[instrument_id - 1],
                        linewidth=5,
                        solid_capstyle='butt'
                    )
                    
                    # Skip ahead
                    i = end_idx + 1
                else:
                    i += 1
    
    # Add instrument range indicators
    midi_values = [note_to_midi(note) for note in notes]
    
    # Find indices corresponding to range boundaries
    flute_low_idx = next((i for i, m in enumerate(midi_values) if m >= FLUTE_RANGE[0]), None)
    flute_high_idx = next((i for i, m in enumerate(midi_values) if m > FLUTE_RANGE[1]), len(midi_values))
    clarinet_low_idx = next((i for i, m in enumerate(midi_values) if m >= CLARINET_RANGE[0]), None)
    clarinet_high_idx = next((i for i, m in enumerate(midi_values) if m > CLARINET_RANGE[1]), len(midi_values))
    
    # Add range indicators
    if flute_low_idx is not None:
        plt.axhline(y=flute_low_idx, color='tab:blue', linestyle='--', alpha=0.5)
        plt.axhline(y=flute_high_idx, color='tab:blue', linestyle='--', alpha=0.5)
    if clarinet_low_idx is not None:
        plt.axhline(y=clarinet_low_idx, color='tab:green', linestyle='--', alpha=0.5)
        plt.axhline(y=clarinet_high_idx, color='tab:green', linestyle='--', alpha=0.5)
    
    # Add labels and title
    plt.xlabel('Time (s)')
    plt.ylabel('Note')
    plt.title('Piano Roll View (One Note Per Instrument with Playable Ranges)')
    
    # Set y-axis limits
    plt.ylim(-0.5, n_notes - 0.5)
    
    # Add y-tick labels (note names) - SHOW ALL NOTES
    plt.yticks(
        np.arange(n_notes),
        notes
    )
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='tab:blue', lw=4, label='Flute'),
        plt.Line2D([0], [0], color='tab:green', lw=4, label='Clarinet'),
        plt.Line2D([0], [0], color='tab:blue', linestyle='--', alpha=0.5, label='Flute Range'),
        plt.Line2D([0], [0], color='tab:green', linestyle='--', alpha=0.5, label='Clarinet Range')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Add grid
    plt.grid(True, which='both', linestyle='--', alpha=0.3)
    
    # Save or display
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        return output_path
    else:
        plt.tight_layout()
        plt.show()
        return None

def main(audio_pkl_path='audio.pkl', ref_db_path='reference_db.pkl'):
    """
    Main function to run the instrument classification process.
    
    Args:
        audio_pkl_path: Path to the processed audio data
        ref_db_path: Path to the reference database
        
    Returns:
        Dictionary of output paths
    """
    # Create output directory
    output_dir = os.path.dirname(audio_pkl_path)
    if not output_dir:
        output_dir = 'output'
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"[test] Loading audio data from {audio_pkl_path}")
    try:
        data = load_audio_data(audio_pkl_path)
    except Exception as e:
        print(f"[test] Error loading audio data: {e}")
        return None
    
    notes = data['notes']
    mag_matrix = data['mag_matrix']
    times = data['times']
    
    print(f"[test] Loading reference database from {ref_db_path}")
    ref_db = load_reference_db(ref_db_path)
    
    # Classify the audio data using range-based classification
    print("[test] Classifying instruments...")
    class_map = classify_events_with_ranges(data, ref_db)
    
    # Create output paths
    outputs = {}
    
    # Plot classification map
    print("[test] Plotting classification map...")
    class_map_path = os.path.join(output_dir, 'classification_map.png')
    outputs['classification_map'] = plot_classification(notes, times, class_map, class_map_path)
    
    # Plot magnitude data
    print("[test] Plotting magnitude data...")
    mag_path = os.path.join(output_dir, 'magnitude.png')
    outputs['magnitude'] = plot_magnitude(notes, times, mag_matrix, mag_path)
    
    # Plot piano roll
    print("[test] Plotting piano roll...")
    piano_roll_path = os.path.join(output_dir, 'piano_roll.png')
    outputs['piano_roll'] = plot_piano_roll(notes, times, class_map, piano_roll_path)
    
    print("[test] Analysis complete. Results saved to:")
    for key, path in outputs.items():
        print(f"  - {key}: {path}")
    
    return outputs

if __name__ == '__main__':
    import sys
    
    # Check if arguments are provided
    if len(sys.argv) > 1:
        audio_pkl_path = sys.argv[1]
    else:
        audio_pkl_path = 'audio.pkl'
        
    if len(sys.argv) > 2:
        ref_db_path = sys.argv[2]
    else:
        ref_db_path = 'reference_db.pkl'
    
    main(audio_pkl_path, ref_db_path)