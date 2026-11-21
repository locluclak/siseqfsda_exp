import matplotlib.pyplot as plt
import os
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))

# --- Global Font Settings ---
# This sets a base size for all text elements in the plot
plt.rcParams.update({'font.size': 14}) 

# Configuration
data_dir = 'exp'
n_values = [40, 80, 120, 160]
categories = ['new', 'old']

results = {'new': [], 'old': []}

def get_mean_runtime(file_path):
    """Reads a file and returns the mean of the values found on each line."""
    try:
        with open(file_path, 'r') as f:
            times = [float(line.strip()) for line in f if line.strip()]
            return np.mean(times) if times else 0
    except FileNotFoundError:
        print(f"Warning: {file_path} not found.")
        return None

# Data Collection
for n in n_values:
    for cat in categories:
        file_name = f"v2time_{cat}_ns{n}.txt"
        file_path = os.path.join(script_dir, file_name)

        path = os.path.join(data_dir, file_name)
        mean_val = get_mean_runtime(path)
        results[cat].append(mean_val)

# Plotting
plt.figure(figsize=(8, 6))
custom_palette = ["#3498db", "#ff8800"]  # Custom colors for the boxplots

plt.plot(n_values, results['new'], marker='o', markersize=10, linestyle='-', linewidth=3, 
         label='Parametric LP approach', color=custom_palette[0])

plt.plot(n_values, results['old'], marker='s', markersize=10, linestyle='-', linewidth=3, 
         label='Independent LP approach', color=custom_palette[1])

# Labels with explicit large font sizes
plt.xlabel('# source instances', fontsize=16, labelpad=15)
plt.ylabel('Average runtime (s)', fontsize=16, labelpad=15)

# Tick marks size
plt.xticks(n_values, fontsize=14)
plt.yticks(fontsize=14)

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=14, frameon=True)

plt.tight_layout()
plt.savefig('exp/comparison_cost_changing_ns.pdf', bbox_inches='tight') # bbox_inches ensures no text is cut off
plt.show()