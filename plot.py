import matplotlib.pyplot as plt
import os
import numpy as np

# --- Global Font Settings ---
# This sets a base size for all text elements in the plot
plt.rcParams.update({'font.size': 16}) 

# Configuration
data_dir = 'exp'
n_values = [30, 40, 50, 60]
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
        file_name = f"time_{cat}_{n}.txt"
        path = os.path.join(data_dir, file_name)
        mean_val = get_mean_runtime(path)
        results[cat].append(mean_val)

# Plotting
plt.figure(figsize=(10, 8))

plt.plot(n_values, results['new'], marker='o', markersize=10, linestyle='-', linewidth=3, 
         label='Parametric Simplex approach', color='#2ecc71')

plt.plot(n_values, results['old'], marker='s', markersize=10, linestyle='-', linewidth=3, 
         label='Discrete Optimization approach', color='#e74c3c')

# Labels with explicit large font sizes
plt.xlabel('# source instances', fontsize=20, labelpad=15)
plt.ylabel('Average runtime (s)', fontsize=20, labelpad=15)

# Tick marks size
plt.xticks(n_values, fontsize=16)
plt.yticks(fontsize=16)

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=16, frameon=True)

plt.tight_layout()
plt.savefig('comparison_cost.pdf', bbox_inches='tight') # bbox_inches ensures no text is cut off
plt.show()