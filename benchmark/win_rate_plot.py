import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import seaborn as sns

fig_size = [30, 15]
font_size = 70
barWidth = 0.7  # Reduced bar width to match reference
scale_x_axis = 1.0

def init_plotting():
    sns.set_style("white")
    plt.rcParams["figure.figsize"] = fig_size  # Adjusted figure size to match reference
    # font_size = 50  # Reduced font size to match reference

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = font_size
    plt.rcParams["axes.labelsize"] = font_size
    plt.rcParams["axes.titlesize"] = font_size
    plt.rcParams["legend.fontsize"] = font_size
    plt.rcParams["xtick.labelsize"] = font_size
    plt.rcParams["ytick.labelsize"] = font_size

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.weight': 'bold',  # Changed to normal weight to match reference
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold'
    })


init_plotting()

data = {
    ('Human', 'Stage 3'): {-5: 11, -4: 27, -3: 24, -2: 9, -1: 2, 0: 0, 1: 1, 2: 7, 3: 10, 4: 11, 5: 0},
    ('Vanilla', 'Stage 3'): {-5: 3, -4: 23, -3: 25, -2: 13, -1: 11, 0: 0, 1: 7, 2: 11, 3: 15, 4: 12, 5: 0},
    ('Vanilla', 'Human'): {-5: 2, -4: 12, -3: 5, -2: 6, -1: 1, 0: 0, 1: 1, 2: 5, 3: 12, 4: 28, 5: 13}
}

# Calculate and print win rates
for comparison, scores in data.items():
    total_votes = sum(scores.values())
    win_votes = sum(scores[k] for k in scores if k > 0)
    lose_votes = sum(scores[k] for k in scores if k < 0)
    win_rate = (win_votes / total_votes) * 100
    lose_rate = (lose_votes / total_votes) * 100

    print(f"\nComparison: {comparison[0]} vs {comparison[1]}")
    print(f"Win Rate for {comparison[0]}: {win_rate:.1f}%")
    print(f"Win Rate for {comparison[1]}: {lose_rate:.1f}%")
    print(f"Total votes: {total_votes}")

detailed_grouped_data = {comparison: {'lose': {k: v for k, v in votes.items() if k < 0},
                                      'win': {k: v for k, v in votes.items() if k > 0}}
                         for comparison, votes in data.items()}

# Split the labels into two parts
# left_labels = ['Vanilla', 'Vanilla', 'Human']
# right_labels = ['Human', 'AI Realtor', 'AI Realtor']
left_labels = list(reversed(['Human\n(lose)', 'Vanilla\n(lose)', 'Human\n(lose)']))
right_labels = list(reversed(['AI Realtor\n(win)', 'AI Realtor\n(win)', 'Vanilla\n(win)']))

# Use colors matching the reference image
# colors_lose = ['#fb6a4a', '#fc9272', '#fcbba1', '#fee0d2', '#fff5f0'][::-1]
# color_map_lose = {-5: colors_lose[0], -4: colors_lose[1], -3: colors_lose[2], -2: colors_lose[3], -1: colors_lose[4]}
#
# colors_win = ['#08519c', '#2171b5', '#4292c6', '#6baed6', '#9ecae1'][::-1]
# color_map_win = {1: colors_win[-1], 2: colors_win[-2], 3: colors_win[-3], 4: colors_win[-4], 5: colors_win[-5]}

colors = ['#fff5f0','#fee0d2','#fcbba1','#fc9272','#fb6a4a','#ef3b2c','#cb181d','#a50f15','#67000d'][1:6][::-1] #['#fee5d9','#fcae91','#fb6a4a','#de2d26','#a50f15'][::-1] #['#fef0d9','#fdd49e','#fdbb84','#fc8d59','#ef6548','#d7301f','#990000']
color_map_lose = {-5: colors[0], -4: colors[1], -3: colors[2], -2: colors[3], -1: colors[4]}

colors = ['#f7fbff','#deebf7','#c6dbef','#9ecae1','#6baed6','#4292c6','#2171b5','#08519c','#08306b'][1:6][::-1]#['#edf8e9','#bae4b3','#74c476','#31a354','#006d2c'][::-1]
color_map_win = {1: colors[-1], 2: colors[-2], 3: colors[-3], 4: colors[-4], 5: colors[-5]}

portion_grouped_data = {
    comparison: {key: sum(votes.values()) for key, votes in detailed_grouped_data[comparison].items()}
    for comparison in detailed_grouped_data}

fig, ax1 = plt.subplots(figsize=fig_size)
plt.rcParams["font.size"] = font_size
ax2 = ax1.twinx()  # Create a twin axis
comparison_keys = list(reversed([("Human", "Stage 3"), ("Vanilla", "Stage 3"), ("Vanilla", "Human")]))

# for idx, (comparison, results) in enumerate(detailed_grouped_data.items()):
for idx, comparison in enumerate(comparison_keys):
    total_votes = portion_grouped_data[comparison]['lose'] + portion_grouped_data[comparison]['win']
    results = detailed_grouped_data[comparison]
    if comparison == ("Vanilla", "Human"):
        # for visualization, we need to essentially flip this results, but originally, we also need to flip the results due to the way the data is stored
        # so in the end, we don't need to flip the results
        # _results = {"lose": results["win"], "win": results["lose"]}
        _results = results
    else:
        _results = {"lose": {-k: v for k, v in results["win"].items()}, "win": {-k: v for k, v in results["lose"].items()}}

    left = 0
    for score, count in sorted(_results['lose'].items()):
        if count > 0:
            portion = count / total_votes * 100
            ax1.barh(idx, portion, left=left, color=color_map_lose[score], height=barWidth)
            left += portion / scale_x_axis

    for score, count in sorted(_results['win'].items()):
        if count > 0:
            portion = count / total_votes * 100
            ax1.barh(idx, portion, left=left, color=color_map_win[score], height=barWidth)
            left += portion / scale_x_axis

# Set up the left y-axis
ax1.set_yticks(range(len(left_labels)))
ax1.set_yticklabels(left_labels)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Set up the right y-axis
ax2.set_yticks(range(len(right_labels)))
ax2.set_yticklabels(right_labels)
ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)

# Align the two axes
ax1.set_ylim(-0.3, 2.3)
ax2.set_ylim(-0.3, 2.3)

# X-axis settings
ax1.set_xlabel('Percentages (%)')
ax1.set_xticks([0, 50, 100])
ax1.set_xlim(0, 100/scale_x_axis)

# Create legend
legend_patches = [
    mpatches.Patch(color=color_map_lose[-5], label='Strong Lose'),
    mpatches.Patch(color=color_map_lose[-1], label='Slight Lose'),
    mpatches.Patch(color=color_map_win[1], label='Slight Win'),
    mpatches.Patch(color=color_map_win[5], label='Strong Win'),
]

# Position legend above the plot, outside the bars
ax1.legend(handles=legend_patches, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.35))

# plt.tight_layout()
plt.savefig("comparison_win_rates_improved.pdf", bbox_inches='tight', dpi=300)