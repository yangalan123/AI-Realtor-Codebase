import csv
import re
import json
import glob
import os.path
import random
import re
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from Levenshtein import distance
from datasets import load_dataset

from utils import load_schema, load_hf_dataset


def convert_response_to_binary(response):
    response = response.strip().lower()
    if response.startswith("yes"):
        return 1
    else:
        return 0


if __name__ == '__main__':
    root_dir = "path/to/dir/highlight_extraction/highlight_model_Meta-Llama-3.1-70B-Instruct_extracted_features_all_data_v10_claude_schema_after_human_annotation_top_10000"
    pattern = f"{root_dir}/ckpt.pt.*"
    schema_dict = load_schema("claude_output/claude_merged_after_baseline_fix_0830.json")
    files = glob.glob(pattern)
    random.seed(1)
    files = [x for x in files if "deprecated" not in x]
    # extract index of ckpts from the filenames -- e.g., highlight_model_extracted_features_all_data_v7_claude_schema.pt.0
    indices = [int(x.split(".")[-1]) for x in files]
    # sort files based on the indices
    files = [x for _, x in sorted(zip(indices, files))]
    outputs = []
    prompts = []
    num_classes = None
    response_text_set = set()
    # extract top_k from root_dir using regex
    top_k = int(re.search(r"top_(\d+)", root_dir).group(1))
    train_dataset = load_hf_dataset("[hf_path]", rating_by_score=True, top_k=top_k,)
    descriptions = train_dataset.to_pandas()['description']
    categories = list(schema_dict.keys())
    final_ckpt_name = os.path.join(root_dir, "checkpoint_merged.pt")
    if not os.path.exists(final_ckpt_name):
        for file in files:
            slice_prompts, slice_outputs = torch.load(file)
            print("Loaded {} outputs from {}".format(len(slice_outputs), file))
            assert len(slice_outputs) % len(
                slice_prompts) == 0, "Number of outputs should be divisible by number of prompts."
            prompts.extend(slice_prompts)
            outputs.extend(slice_outputs)
            for output_i, output in enumerate(slice_outputs):
                response_text_set |= set([x.strip().lower() for x in output["outputs"]])
                # compute levenstein distance between the response prompt and the original prompt
                try:
                    assert output['prompt'] == slice_prompts[
                        output_i], "Prompt mismatch at {}:\noutput.prompt:\n{} \noriginal_prompt:\n{}".format(output_i,
                                                                                                              output[
                                                                                                                  'prompt'],
                                                                                                              prompts[
                                                                                                                  output_i])
                except Exception as e:
                    print(e)
                    dist = distance(output['prompt'].strip().lower(), slice_prompts[output_i].strip().lower())
                    assert dist / len(prompts[output_i]) < 0.1, "fuzzy match failed, dist: {}/{}".format(dist, len(
                        slice_prompts[output_i]))
        print("Total number of prompts: {}".format(len(prompts)))
        print("Total number of outputs: {}".format(len(outputs)))
        print("Total responses variants: {}".format(random.sample(list(response_text_set), 5)))
        assert len(outputs) % len(
            descriptions) == 0, "Number of outputs ({}) should be divisible by number of descriptions ({}).".format(
            len(outputs), len(descriptions))
        num_classes = len(outputs) // len(descriptions)
        print("Number of classes: {}".format(num_classes))
        processed_dataset = []
        for idx in range(len(descriptions)):
            dataset_idx = idx
            description = descriptions[dataset_idx]
            datapoint = {"description": description, "classes": []}
            assert train_dataset[dataset_idx][
                       'description'] == description, "Description mismatch at idx: {}\ndescription:\n{}\ntrain_dataset:\n{}".format(
                idx, description, train_dataset[dataset_idx]['description'])
            datapoint['id'] = train_dataset[dataset_idx]['id']
            for class_idx in range(idx * num_classes, (idx + 1) * num_classes):
                assert description in prompts[
                    class_idx], "Description mismatch at class_idx: {}\ndescription:\n{}\n---prompt:-----\n{}\n---previous prompt:---\n{}\n---next prompt:---\n{}".format(
                    class_idx,
                    description,
                    prompts[
                        class_idx], prompts[class_idx - 1], prompts[class_idx + 1])
                _outputs = outputs[class_idx]["outputs"]
                _binaries = [convert_response_to_binary(x) for x in _outputs]
                # class_name_candidates = []
                # Extract category from the prompt
                # in format: "Provided Highlight Category: {}\n"
                # only extract {} part, using regex
                # match = re.search(r"Provided Highlight Category: (.+?)\n", prompts[class_idx])
                match = re.search(r"Label Class Category: (.+?)\n", prompts[class_idx])
                assert match, "No match found in prompt: {}".format(prompts[class_idx])
                category = match.group(1).strip()[:-1]
                assert category in categories, "Category not found in schema: '{}'".format(category)
                # for category in categories:
                #     if category in prompts[class_idx]:
                #         class_name_candidates.append(category)
                # assert len(class_name_candidates) == 1, "Class name candidates: {}".format(class_name_candidates)
                # majority voting (if >=1) to get the final class
                _binaries_counter = sum(_binaries)
                if _binaries_counter > len(_binaries) // 2:
                    datapoint["classes"].append(category)
            processed_dataset.append(datapoint)

        torch.save((prompts, outputs, response_text_set, processed_dataset, num_classes),
                   final_ckpt_name)
    else:
        prompts, outputs, response_text_set, processed_dataset, num_classes = torch.load(final_ckpt_name)

    unigram_counter = Counter()
    bigram_counter = Counter()
    stat_dict = defaultdict(list)
    for datum in processed_dataset:
        data_classes = set(datum['classes'])
        assert len(data_classes) == len(datum['classes']), "Duplicate classes found in datum: {}".format(datum)
        for class_name in data_classes:
            unigram_counter[class_name] += 1
        stat_dict["num_of_classes_per_instance"].append(len(data_classes))
        data_classes = list(data_classes)
        for i in range(len(data_classes)):
            for j in range(i + 1, len(data_classes)):
                bigram_counter[(data_classes[i], data_classes[j])] += 1
                bigram_counter[(data_classes[j], data_classes[i])] += 1
    print("Avg number of classes per instance: {}".format(np.mean(stat_dict["num_of_classes_per_instance"])))
    # visualization_dir = "highlight_model_claude_schema_class_visualization"
    visualization_dir = os.path.join(root_dir, "visualization")
    os.makedirs(visualization_dir, exist_ok=True)
    # compute overall keyword containing rate
    keyword_containing_rate = 0
    all_classes_num = 0
    for datum in processed_dataset:
        data_classes = datum['classes']
        for _class in data_classes:
            keyword_containing_rate += any([x.lower() in datum['description'].lower() for x in schema_dict[_class]])
            all_classes_num += 1
    print("Keyword containing rate: {:.3f}".format(keyword_containing_rate / all_classes_num))
    # among all classes, compute the rate that we can use keyword matching to determine the class, but perhaps the model fails
    all_data_keyword_matching_rate = 0
    keyword_match_failure_cases = []
    for datum in processed_dataset:
        data_classes = datum['classes']
        # note here we have to iterate over all classes, since the model may fail to predict some classes that can be simply determined by keyword matching
        for _class in schema_dict:
            judges = [x.lower() in datum['description'].lower() for x in schema_dict[_class]]
            matched_keywords = []
            for judge_i, judge in enumerate(judges):
                if judge:
                    matched_keywords.append(schema_dict[_class][judge_i])
            flag = any(judges)
            all_data_keyword_matching_rate += flag
            if flag and _class not in data_classes:
                new_failure_case = {
                    "id": datum['id'],
                    "description": datum['description'],
                    "fail_to_match_class": _class,
                    "class_keywords": schema_dict[_class],
                    "matched_keywords": matched_keywords
                }
                keyword_match_failure_cases.append(new_failure_case)
    with open(os.path.join(root_dir, "keyword_matching_failure_cases.json"), "w") as f_out:
        json.dump(keyword_match_failure_cases, f_out, indent=4)

    print("All data keyword matching rate: {:.3f}".format(keyword_containing_rate / all_data_keyword_matching_rate))
    sample_num = 300
    # sample some outputs for manual verification
    sampled_dataset = random.sample(processed_dataset, sample_num)
    with open(os.path.join(visualization_dir, "sampled_dataset_instancewise.csv"), "w") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=["id", "description", "class", "class_keywords",
                                                   "keyword_contained_in_desc"])
        writer.writeheader()
        buf = []
        for datum in sampled_dataset:
            data_classes = datum['classes']
            for _class in data_classes:
                flag =  any(
                        [x.lower() in datum['description'].lower() for x in schema_dict[_class]])
                if not flag:
                    new_datapoint = {
                        "id": datum['id'],
                        "description": datum['description'],
                        "class": _class,
                        "class_keywords": schema_dict[_class],
                        "keyword_contained_in_desc": flag
                    }
                    buf.append(new_datapoint)
        writer.writerows(buf)
    classwise_sample_num = 5
    with open(os.path.join(visualization_dir, "sampled_dataset_classwise.csv"), "w") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=["id", "description", "class", "class_keywords",
                                                   "keyword_contained_in_desc"])
        writer.writeheader()
        buf = []
        class_dict = defaultdict(list)
        for datum in sampled_dataset:
            data_classes = datum['classes']
            for _class in data_classes:
                flag =  any(
                    [x.lower() in datum['description'].lower() for x in schema_dict[_class]])
                if not flag:
                    new_datapoint = {
                        "id": datum['id'],
                        "description": datum['description'],
                        "class": _class,
                        "class_keywords": schema_dict[_class],
                        "keyword_contained_in_desc": flag
                    }
                    class_dict[_class].append(new_datapoint)
        for _class in class_dict:
            # sample 10 instances for each class
            sampled_instances = random.sample(class_dict[_class], min(10, len(class_dict[_class])))
            buf.extend(sampled_instances)
        writer.writerows(buf)

    os.makedirs(visualization_dir, exist_ok=True)
    # task 1: draw histogram of classes
    class_names = list(unigram_counter.keys())
    class_values = [unigram_counter[x] for x in class_names]
    plt.bar(class_names, class_values)
    plt.xticks(rotation=90)
    # set the title
    plt.title("Histogram of Classes")
    # save the plot
    plt.savefig(os.path.join(visualization_dir, "histogram_of_classes.png"))
    # clean
    plt.clf()
    # task 2: draw pmi of classes
    class_names = list(unigram_counter.keys())
    # compute pairwise pmi
    # compute unigram normalized distribution
    unigram_distribution = dict()
    # total_unigram = sum(unigram_counter.values())
    unigram_normalizer = len(processed_dataset)
    for class_name in class_names:
        unigram_distribution[class_name] = unigram_counter[class_name] / unigram_normalizer
    # compute bigram normalized distribution
    bigram_distribution = dict()
    # total_bigram = sum(bigram_counter.values())
    bigram_normalizer = len(processed_dataset)
    for bigram in bigram_counter:
        bigram_distribution[bigram] = bigram_counter[bigram] / bigram_normalizer
    pmi_values = dict()
    for i in range(len(class_names)):
        for j in range(i + 1, len(class_names)):
            if (class_names[i], class_names[j]) in bigram_distribution:
                pmi = np.log(bigram_distribution[(class_names[i], class_names[j])])
            else:
                pmi = 0
            pmi = pmi - np.log(unigram_distribution[class_names[i]]) - np.log(unigram_distribution[class_names[j]])
            pmi_values[(class_names[i], class_names[j])] = pmi
    pmi_items = list(pmi_values.items())
    pmi_items.sort(key=lambda x: abs(x[1]), reverse=True)
    print("Top 10 PMI values:")
    for i in range(10):
        print(pmi_items[i], unigram_distribution[pmi_items[i][0][0]], unigram_distribution[pmi_items[i][0][1]])
    print("Smallest 10 PMI values:")
    for i in range(10):
        print(pmi_items[-i - 1], unigram_distribution[pmi_items[-i - 1][0][0]],
              unigram_distribution[pmi_items[-i - 1][0][1]])
    print("Top unigram stat:")
    unigram_items = list(unigram_distribution.items())
    unigram_items.sort(key=lambda x: x[1], reverse=True)
    for i in range(10):
        print("{:.3f}".format(unigram_items[i][1]), unigram_items[i][0])
    print("Least unigram stat:")
    unigram_items.sort(key=lambda x: x[1], reverse=False)
    for i in range(10):
        print("{:.3f}".format(unigram_items[i][1]), unigram_items[i][0])
    print("Top #(A and B)^2/(A)(B):")
    binary_stats = []
    for i in range(len(class_names)):
        for j in range(i + 1, len(class_names)):
            _stat = bigram_counter[(class_names[i], class_names[j])] ** 2 / (
                    unigram_counter[class_names[i]] * unigram_counter[class_names[j]])
            binary_stats.append((_stat, class_names[i], class_names[j]))
    binary_stats.sort(key=lambda x: x[0], reverse=True)
    for i in range(10):
        # for 0-th element in binary_stat, keep only 3 decimal points
        print("{:.3f}".format(binary_stats[i][0]), binary_stats[i][1], binary_stats[i][2])
    # temperature-related features
    # print("Temperature-related features:")
    stat_single_side_conditionals = []
    for i in range(len(class_names)):
        for j in range(i + 1, len(class_names)):
            _stat_0 = bigram_counter[(class_names[i], class_names[j])] / unigram_counter[class_names[i]]
            _stat_1 = bigram_counter[(class_names[i], class_names[j])] / unigram_counter[class_names[j]]
            stat_single_side_conditionals.append((max(_stat_0, _stat_1), class_names[i], class_names[j]))

    stat_single_side_conditionals.sort(key=lambda x: x[0], reverse=True)
    print("Top correlated features (#(AB)/#(A) | #(AB)/#(A)):")
    for i in range(len(stat_single_side_conditionals)):
        if stat_single_side_conditionals[i][0] < 0.75:
            break
        print("{}||{}||{:.3f}".format(stat_single_side_conditionals[i][1], stat_single_side_conditionals[i][2],
                                      stat_single_side_conditionals[i][0]))

    # plot heatmap
    pmi_matrix = np.zeros((len(class_names), len(class_names)))
    for i in range(len(class_names)):
        for j in range(i + 1, len(class_names)):
            pmi_matrix[i, j] = abs(pmi_values[(class_names[i], class_names[j])])
            pmi_matrix[j, i] = abs(pmi_values[(class_names[i], class_names[j])])
    plt.imshow(pmi_matrix, cmap='hot', interpolation='nearest')
    # set the title
    plt.title("Pointwise Mutual Information (absolute value) of Classes")
    # set the legend
    plt.colorbar()
    # save the plot
    plt.savefig(os.path.join(visualization_dir, "pmi_of_classes.png"))
    plt.clf()
    # task 3: plot the histogram of number of classes per instance
    plt.hist(stat_dict["num_of_classes_per_instance"], bins=range(0, num_classes))
    # set the title
    plt.title("Histogram of Number of Classes per Instance")
    # save the plot
    plt.savefig(os.path.join(visualization_dir, "histogram_of_num_classes_per_instance.png"))
