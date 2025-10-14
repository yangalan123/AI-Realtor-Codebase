import os

from utils import get_highlight_data
from tqdm import tqdm
import torch
from const import DESIRED_FEATURE_NAMES, desired_to_originals
from collections import Counter
from utils import judge_empty_value

if __name__ == '__main__':
    all_features, extracted_data_origin = get_highlight_data()
    ids_origin = list(extracted_data_origin.keys())
    ckpt_file_name = "zillow_feature_embedding.ckpt"
    extracted_data, counter, ids, processed_ids = torch.load(ckpt_file_name)
    ids = list(extracted_data.keys())
    assert set(ids) == set(ids_origin), "The ids in the embedding file and the original data file are not the same. Delta Size: {}".format(len(set(ids) ^ set(ids_origin)))
    label_counter = Counter()
    label_num_counter = Counter()
    label_num_to_description = dict()
    label_input_counter = Counter()
    all_not_nan_features = set()
    for _id in tqdm(extracted_data_origin):
        feature_list = list(extracted_data_origin[_id].keys())
        labels = set()
        for key in feature_list:
            # assert extracted_data_origin[_id][key] == extracted_data[_id][key], "The highlighted feature value is not the same. ID: {}, Key: {}, origin: {}, new: {}".format(_id, key, extracted_data_origin[_id][key], extracted_data[_id][key])
            assert key in extracted_data[_id], "The key is not in the new extracted data. ID: {}, Key: {}".format(_id, key)
            # if extracted_data_origin[_id][key] is not None and key in DESIRED_FEATURE_NAMES:
            if judge_empty_value(extracted_data_origin[_id][key]) and key in DESIRED_FEATURE_NAMES:
                labels.add(key)
        all_feature_list = set()
        if "embeddings" in extracted_data[_id]:
            for key in all_features[_id]:
                if all_features[_id][key] is not None:
                    all_feature_list.add(key)
            assert len(all_feature_list) == len(extracted_data[_id]['embeddings']), "The feature number is not the same. ID: {}, All Features: {}, Extracted Features: {}".format(_id, len(all_feature_list), len(extracted_data[_id]['embeddings']))
        all_not_nan_features |= all_feature_list
        if len(all_feature_list) > 0:
            label_input_counter.update(all_feature_list)
        if feature_list != list(extracted_data[_id].keys()):
            delta_set = set(extracted_data[_id].keys()) - set(feature_list)
            for key in delta_set:
                # if key in DESIRED_FEATURE_NAMES and extracted_data[_id][key]:
                if key in DESIRED_FEATURE_NAMES and judge_empty_value(extracted_data[_id][key]):
                    labels.add(key)
        for label in labels:
            label_counter[label] += 1
        label_num_counter[len(labels)] += 1
        if len(labels) not in label_num_to_description:
            label_num_to_description[len(labels)] = []
        label_num_to_description[len(labels)].append(_id)

    torch.save(list(all_not_nan_features), "all_not_nan_features.pt")
    visualization_root_dir = "./visualization"
    os.makedirs(visualization_root_dir, exist_ok=True)
    # convert counter to histogram (frequency)
    # plot the histogram with proper bins
    import matplotlib.pyplot as plt
    plt.figure()
    # first plot the histogram of the number of highlighted features
    all_labels = list(label_num_counter.keys())
    all_labels.sort()
    all_values = [label_num_counter[label] for label in all_labels]
    plt.bar(all_labels, all_values)
    plt.xlabel("Feature Numbers")
    plt.ylabel("Frequency")
    plt.title("Histogram of the number of highlighted features")
    plt.savefig(os.path.join(visualization_root_dir, "highlighted_feature_num_histogram.png"))
    plt.close()
    # then plot the histogram of the highlighted features, in terms of percentage
    all_labels = list(label_counter.keys())
    all_labels.sort()
    all_values = [label_counter[label]/len(extracted_data_origin) for label in all_labels]
    # all_values = [x / sum(all_values) for x in all_values]
    plt.figure()
    plt.bar(all_labels, all_values)
    plt.xlabel("Feature Names")
    plt.ylabel("Frequency")
    plt.title("Histogram of the highlighted features")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(visualization_root_dir, "highlighted_feature_histogram.png"))
    plt.close()
    # then plot the utility of each feature, utility = frequency / input frequency
    all_labels = list(label_counter.keys())
    all_labels.sort()
    all_labels = [x for x in all_labels if label_input_counter[x] > 0]
    all_values = [label_counter[label] / label_input_counter[label] for label in all_labels if label_input_counter[label] > 0]
    plt.figure()
    plt.bar(all_labels, all_values)
    plt.xlabel("Feature Names")
    plt.ylabel("Utility")
    plt.title("Utility of the highlighted features")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(visualization_root_dir, "highlighted_feature_utility.png"))
    plt.close()
    for line in label_num_to_description[20]:
        print(line)

    print(label_counter['is_modern'])





