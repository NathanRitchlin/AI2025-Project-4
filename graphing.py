import matplotlib.pyplot as plt
import numpy as np

# Number of categories and bars per category
n_categories = 6
n_bars = 3

# Example data (replace with your own)
data = [[10,9,21,24,27,42], [11.2,14,36.4,61.2,52,73.6], [14,26,84,135,105,162]]
# Category labels
categories = ["W Track/NRST", "W Track/FRST", "U Track/NRST", "U Track/FRST", "2 Track/NRST", "2 Track/FRST"]

# Bar positions
x = np.arange(n_categories)
width = 0.25  # width of each bar

plt.figure(figsize=(10, 6))
colors = ["green", "yellow", "red"]
groups = ["Fastest", "Average", "Slowest"]
for i in range(n_bars):
    plt.bar(x + i * width, data[i], width=width, color = colors[i], label=groups[i])

plt.xticks(x + width, categories)
plt.xlabel("Track Settings")
plt.ylabel("Race Times")

plt.legend()
plt.tight_layout()
plt.savefig("results/BarGraphs/SARSAResults.png")
