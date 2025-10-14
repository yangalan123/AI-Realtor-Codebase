import matplotlib.pyplot as plt
import numpy as np

# Set the style
plt.style.use("seaborn-v0_8-whitegrid")

# Set base font size
base_font_size = 55

# Configure font settings
plt.rcParams.update({
    'font.size': base_font_size,
    #'font.weight': 'bold',
    #'axes.labelweight': 'bold',
    #'axes.titleweight': 'bold'
})

# Data
name_map = {
    "human": "Human",
    "gpt4o-mini sft": "SFT",
    "none": "Vanilla",
    "highlight + preference + surprisal": "AI Realtor",
}

models = ["human", "gpt4o-mini sft", "none", "highlight + preference + surprisal"]
model_names = [name_map[model] for model in models]

faithful_hard = [81.31, 84.38, 98.85, 99.24]
error_bar_hard = [0.86, 2.14, 0.15, 0.12]
faithful_soft = [87.69, 86.10, 89.16, 89.08]
error_bar_soft = [0.40, 1.09, 0.25, 0.26]

# Create the figure and axis
fig, ax = plt.subplots(figsize=(20, 15))

x = np.arange(2)  # 2 positions for Faithful_hard and Faithful_soft
width = 0.2  # width of each bar
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # distinct colors for each model

# Plot bars for each model
for i, (model, hard, soft, err_hard, err_soft) in enumerate(zip(model_names, faithful_hard, faithful_soft, error_bar_hard, error_bar_soft)):
    ax.bar(x[0] - width*1.5 + i*width, hard, width, label=model, color=colors[i], yerr=err_hard, error_kw={"elinewidth": 3})
    ax.bar(x[1] - width*1.5 + i*width, soft, width, color=colors[i], yerr=err_soft, error_kw={"elinewidth": 3})

# Customize the plot
ax.set_ylabel('Score', fontsize=base_font_size)
# ax.set_title('Model Performance Comparison', fontsize=base_font_size)
ax.set_xticks(x)
ax.set_xticklabels([r'$\mathrm{Faithful}_\mathrm{hard}$', r'$\mathrm{Faithful}_\mathrm{soft}$'], fontsize=base_font_size * 1.3)
ax.tick_params(axis='both', which='major', labelsize=base_font_size)

# Move legend to top-right
ax.legend(fontsize=base_font_size, loc='upper right', bbox_to_anchor=(1, 1), ncol=1)

# Set y-axis range to highlight differences
ax.set_ylim(80, 100)

# Remove vertical grid lines
ax.yaxis.grid(False)
ax.xaxis.grid(False)

# Adjust layout and display
fig.tight_layout()
plt.savefig('model_hallucination_comparison.pdf', format='pdf')
plt.show()
