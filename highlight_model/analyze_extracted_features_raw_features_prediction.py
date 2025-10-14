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
    root_dir = "path/to/dir/highlight_extraction/highlight_model_Meta-Llama-3.1-70B-Instruct_extracted_features_all_data_predict_raw_feature_baseline_top_2000"
    pattern = f"{root_dir}/ckpt.pt.*"
    ckpt_prompt_name = f"{root_dir}/prompts.pt"

    total_prompts, scores, schema_dict, id2range, all_output_features = torch.load(ckpt_prompt_name)
    # schema_dict = load_schema("claude_output/claude_merged.json")
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
    descriptions = train_dataset.to_pandas()['description'].tolist()
    ids = train_dataset.to_pandas()['id'].tolist()
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
        print("Number of classes: {}".format(len(all_output_features)))
        processed_dataset = []
        for idx in range(len(descriptions)):
            dataset_idx = idx
            _id = ids[dataset_idx]
            _range = id2range[_id]
            description = descriptions[dataset_idx]
            datapoint = {"description": description, "classes": []}
            assert train_dataset[dataset_idx][
                       'description'] == description, "Description mismatch at idx: {}\ndescription:\n{}\ntrain_dataset:\n{}".format(
                idx, description, train_dataset[dataset_idx]['description'])
            datapoint['id'] = train_dataset[dataset_idx]['id']
            for class_idx in range(_range[0], _range[1]):
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
                _binaries_counter = sum(_binaries)
                if _binaries_counter > len(_binaries) // 2:
                    datapoint["classes"].append(category)
            processed_dataset.append(datapoint)

        torch.save((prompts, outputs, response_text_set, processed_dataset, num_classes),
                   final_ckpt_name)
        print("Saved to {}".format(final_ckpt_name))
    else:
        prompts, outputs, response_text_set, processed_dataset, num_classes = torch.load(final_ckpt_name)
        print("Loaded from {}".format(final_ckpt_name))

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
