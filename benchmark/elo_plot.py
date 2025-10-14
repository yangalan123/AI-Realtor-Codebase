import sqlite3
import pandas as pd
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle


class EloRating:
    def __init__(self, initial_rating=1000, K=32, scale=400, base=10):
        self.ratings = {}
        self.initial_rating = initial_rating
        self.scale = scale
        self.base = base
        self.K = K

    def get_rating(self, player):
        if player not in self.ratings:
            self.ratings[player] = self.initial_rating
        return self.ratings[player]

    def expected_score(self, rating1, rating2):
        return 1 / (1 + self.base ** ((rating2 - rating1) / self.scale))

    def update_ratings(self, winner, loser, score):
        # score = 1

        winner_rating = self.get_rating(winner)
        loser_rating = self.get_rating(loser)

        expected_winner = self.expected_score(winner_rating, loser_rating)
        expected_loser = 1 - expected_winner

        new_winner_rating = winner_rating + self.K * score * (1 - expected_winner)
        new_loser_rating = loser_rating + self.K * score * (0 - expected_loser)

        self.ratings[winner] = new_winner_rating
        self.ratings[loser] = new_loser_rating

    def get_ratings(self):
        return self.ratings

### load rating from pickle

with open('ratings.pkl', 'rb') as f:
    elo = pickle.load(f)


### rename the model name
# name_map = {
#     "basic": "Control",
#     "human": "Human",
#     "none": "GPT-4o",
#     "highlight": "Stage 1",
#     # "highlight + preference": "Stage 2",
#     "highlight_raw_preference": "Stage 2",
#     "highlight + preference + surprisal": "Stage 3",
#     # "gpt4o-mini sft": "SFT",
# }
fontsize = 50
figsize = (60, 30)
linewidth = 5

name_map = {
    "basic gpt-4o": "Control",
    "human gpt-4o": "Human",
    "none gpt-4o-mini": "GPT-4o-mini",
    "none gpt-4o": "GPT-4o",
    # "highlight gpt-4o-mini": "HighlightOnly GPT-4o-mini",
    "highlight gpt-4o-mini": "HighlightOnly GPT-4o-mini",
    "highlight gpt-4o": "HighlightOnly GPT-4o",
    "highlight_raw_preference gpt-4o-mini": "NoSurprisal GPT-4o-mini",
    "highlight_raw_preference gpt-4o": "NoSurprisal GPT-4o",
    "highlight + preference + surprisal gpt-4o-mini": "AI Realtor GPT-4o-mini",
    "highlight + preference + surprisal gpt-4o": "AI Realtor GPT-4o",
    "gpt4o-mini sft gpt-4o": "SFT GPT-4o-mini",
}

ratings = {}
for model in name_map:
    ratings[name_map[model]] = elo.get_rating(model)

# Prepare the data
model_groups = {
    "Control": ["Control"],
    "Human": ["Human"],
    "SFT": ["SFT GPT-4o-mini"],
    "Vanilla": ["GPT-4o", "GPT-4o-mini"],
    # "Stage 1": ["Stage 1 GPT-4o", "Stage 1 GPT-4o-mini"],
    # "Stage 2": ["Stage 2 GPT-4o", "Stage 2 GPT-4o-mini"],
    # "Stage 3": ["Stage 3 GPT-4o", "Stage 3 GPT-4o-mini"],
    # "HighlightOnly": ["HighlightOnly GPT-4o", "HighlightOnly GPT-4o-mini"],
    "AI Realtor [Only Grounding]": ["HighlightOnly GPT-4o", "HighlightOnly GPT-4o-mini"],
    # "NoSurprisal": ["NoSurprisal GPT-4o", "NoSurprisal GPT-4o-mini"],
    "AI Realtor [w/o Marketing]": ["NoSurprisal GPT-4o", "NoSurprisal GPT-4o-mini"],
    "AI Realtor": ["AI Realtor GPT-4o", "AI Realtor GPT-4o-mini"],
}

# Set the style for a more professional look
# plt.style.use("seaborn-whitegrid")
plt.style.use("seaborn-v0_8-whitegrid")

# Configure font settings
plt.rcParams.update({
    'font.size': fontsize,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold'
})

# Create the figure and axis
fig, ax = plt.subplots(figsize=figsize, dpi=300)  # Increased figure size

# Define color palette
base_colors = sns.color_palette("pastel", len(model_groups))

# Plot grouped bars
x = 2 * np.arange(len(model_groups))
width = 1.6  # Increased overall width
bar_width = width / 2  # Wider individual bars

legend_handles = []

for i, (group, models) in enumerate(model_groups.items()):
    group_ratings = [ratings[model] for model in models]
    num_models = len(models)
    
    # Calculate positions for bars within the group
    positions = x[i] + np.linspace(-(num_models-1)*bar_width/2, (num_models-1)*bar_width/2, num_models)
    
    # Create slightly different colors for bars within the same group
    group_colors = [base_colors[i]] * num_models
    if num_models > 1:
        group_colors[0] = sns.set_hls_values(base_colors[i], l=.5)  # Darken first bar
        group_colors[1] = sns.set_hls_values(base_colors[i], l=.7)  # Lighten second bar
    
    for j, (position, rating, model) in enumerate(zip(positions, group_ratings, models)):
        rect = ax.bar(position, rating, bar_width, color=group_colors[j])
        
        # Add to legend
        if "GPT-4o-mini" in model:
            legend_handles.append((rect[0], f"{group} (GPT-4o-mini)"))
        elif "GPT-4o" in model:
            legend_handles.append((rect[0], f"{group} (GPT-4o)"))
        else:
            legend_handles.append((rect[0], group))
        
        # Add value labels on top of each bar
        height = rect[0].get_height()
        ax.annotate(f'{height:.0f}',
                    xy=(rect[0].get_x() + rect[0].get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=int(fontsize))

# Customize the plot
# ax.set_xlabel('Models', fontsize=fontsize, fontweight='bold')
ax.set_ylabel('Elo Ratings', fontsize=fontsize, fontweight='bold')
# ax.set_title('Model Performance Comparison using Elo Ratings', fontsize=16, fontweight='bold', pad=20)

# Set x-axis ticks
ax.set_xticks(x)
# ax.set_xticklabels(model_groups.keys(), rotation=45, ha='right')
keys = list(model_groups.keys())
# replace the first space to be \n
# xticks = [key.replace(" ", "\n", 1) for key in keys]
xticks_mapping = {
    "Control": "Control",
    "Human": "Human",
    "SFT": "SFT",
    "Vanilla": "Vanilla",
    "AI Realtor [Only Grounding]": "AI Realtor\n[Only Grounding]",
    "AI Realtor\n[Only Grounding]": "AI Realtor\n[Only Grounding]",
    "AI Realtor [w/o Marketing]": "AI Realtor\n[w/o Marketing]",
    "AI Realtor\n[w/o Marketing]": "AI Realtor\n[w/o Marketing]",
    "AI Realtor": "AI Realtor",
}
xticks = [xticks_mapping[key] for key in keys]
ax.set_xticklabels(xticks)

# Draw a line for initial Elo ratings
initial_rating_line = ax.axhline(y=1000, color='#FF9999', linestyle='--', linewidth=linewidth)
legend_handles.append((initial_rating_line, 'Initial Rating'))

# Add legend
ax.legend(*zip(*legend_handles), fontsize=fontsize, loc='upper left', bbox_to_anchor=(1, 1))

# Set y-axis limits to give some space above the highest bar
ax.set_ylim(0, max(ratings.values()) * 1.1)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Adjust layout and save
plt.tight_layout()
plt.savefig('elo_ratings_grouped.pdf', bbox_inches='tight')
plt.close()


