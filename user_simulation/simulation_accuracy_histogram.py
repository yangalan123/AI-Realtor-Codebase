import torch
import matplotlib.pyplot as plt

if __name__ == '__main__':

    accuracy_filename = "accuracy.pt.gpt-4o-mini.offline"
    accuracy = torch.load(accuracy_filename)
    user_accuracy = {k: sum(vv[0] for vv in v.values()) / sum(vv[1] for vv in v.values()) for k, v in accuracy.items()}
    # plot the histogram
    figsize = (28, 23)
    fontsize = 65
    plt.figure(figsize=figsize)
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.weight': 'bold',  # Changed to normal weight to match reference
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold'
    })
    plt.rcParams["font.size"] = fontsize
    plt.hist(user_accuracy.values(), bins=20)
    plt.xlabel("User Simulation Accuracy")
    plt.ylabel("User Count")
    plt.tight_layout()
    plt.savefig("accuracy_histogram.pdf", dpi=300, bbox_inches='tight')
