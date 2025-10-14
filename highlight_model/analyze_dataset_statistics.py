from utils import load_hf_dataset
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    output_root_dir = "dataset_statistics"
    os.makedirs(output_root_dir, exist_ok=True)
    dataset = load_hf_dataset()
    scores = [x for x in dataset['score'] if x is not None]
    # plot histogram of scores
    plt.hist(scores, bins=20)
    plt.xlabel('Scores')
    plt.ylabel('Frequency')
    plt.title('Histogram of Scores')
    # save histogram of scores
    plt.savefig(os.path.join(output_root_dir, "score_histogram.pdf"))
    plt.clf()
    # plot accumulated histogram of scores
    plt.figure()
    plt.hist(scores, bins=20, cumulative=True)
    plt.xlabel('Scores')
    plt.ylabel('Frequency')
    plt.title('Accumulated Histogram of Scores')
    # save accumulated histogram of scores
    plt.savefig(os.path.join(output_root_dir, "score_accumulated_histogram.pdf"))

