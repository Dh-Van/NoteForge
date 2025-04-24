import matplotlib.pyplot as plt
import numpy as np
import pickle


with open("note_matrix.pkl", 'rb') as f:
    (labels, note_list, times) = pickle.load(f)



def plot_notes(note_matrix):
    """
    Plots the note matrix with time on the x-axis, notes on the y-axis,
    and instruments differentiated by color.
    """
    cmap_name="tab10"
    instruments=["Clarinet", "Flute"]
    
    # Generate the color map for instruments
    cmap = plt.get_cmap(cmap_name)
    n = len(instruments)
    colors = {inst: cmap(i / (n - 1) if n > 1 else 0.5) for i, inst in enumerate(instruments)}

    # Create the plot
    plt.figure(figsize=(10, 6))
    for time, row in enumerate(note_matrix):
        for note, instrument in enumerate(row):
            if instrument != "0":  # Check if there is an instrument
                if instrument in colors:
                    plt.scatter(time, note, color=colors[instrument], label=instrument, s=50)

    # Remove duplicate labels in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title="Instruments")

    # Add labels and title
    plt.xlabel("Time")
    plt.ylabel("Note")
    plt.title("Note Matrix Visualization")
    plt.grid(True)
    plt.show()

nm = [["0", "Violin", "0"], ["Flute","0", "0"], ["0", "0", "Violin"]]
plot_notes(nm)

