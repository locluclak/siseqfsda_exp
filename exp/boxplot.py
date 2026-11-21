import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# 1. Load the data from files
with open(os.path.join(script_dir, 'time_new_50.txt'), 'r') as f:
    new_data = [float(line.strip()) for line in f if line.strip()]

with open(os.path.join(script_dir, 'time_old_50.txt'), 'r') as f:
    old_data = [float(line.strip()) for line in f if line.strip()]

# 2. Convert to a DataFrame
df_new = pd.DataFrame({'Time': new_data, 'Method': 'Parametric LP'})
df_old = pd.DataFrame({'Time': old_data, 'Method': 'Independent LP'})
df = pd.concat([df_new, df_old])

# 3. Create the plot
plt.figure(figsize=(8, 6))

# Apply font size 14 globally for this plot
plt.rcParams.update({'font.size': 14}) 
custom_palette = ["#3498db", "#ff8800"]  # Custom colors for the boxplots
ax = sns.boxplot(x='Method', y='Time', data=df, palette=custom_palette)

# Set specific label sizes for clarity
ax.set_xlabel('Method', fontsize=14)
ax.set_ylabel('Time', fontsize=14)
ax.tick_params(labelsize=14)

plt.grid(axis='y', linestyle='-', alpha=0.7)

# Ensure directory exists before saving
os.makedirs('exp', exist_ok=True)
plt.savefig('exp/boxplot_comparison_50.pdf', bbox_inches='tight') 
plt.show()