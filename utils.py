import subprocess
import torch
import json
from const import desired_to_originals, DESIRED_FEATURE_NAMES
import re
from difflib import SequenceMatcher
from datasets import load_dataset

def get_original_all_features_data():
    dataset = load_dataset("Sigma-Lab/AI_Realtor_Listing_Data")['train']
    if not os.path.exists("./data"):
        os.makedirs("./data")
    dataset.to_json("./data/ai_realtor_listing_data.json")
    with open("./data/ai_realtor_listing_data.json", "r") as f_in:
        dataset_json = [json.loads(line) for line in f_in]
    all_features = dict()
    for item in dataset_json:
        _id = item["id"]
        all_features[_id] = item
    return all_features

def judge_empty_value(value):
    if value is None:
        return False
    if isinstance(value, str) and value.strip() == "":
        return False
    if isinstance(value, list) and len(value) == 0:
        return False
    return True

def get_all_not_nan_features():
    filename = "all_not_nan_features.pt"
    return torch.load(filename)


def get_gpu_memory_and_usage_rate(logger):
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True)
        if result.returncode == 0:
            gpu_info = result.stdout
            gpu_info_lines = gpu_info.strip().split("\n")
            gpu_summary = [line.split(", ") for line in gpu_info_lines]
            for gpu in gpu_summary:
                logger.info(f"GPU {gpu[0]}: {gpu[1]}, Total Memory: {gpu[2]} MB, Used Memory: {gpu[3]} MB, Free Memory: {gpu[4]} MB, Utilization: {gpu[5]}%")
        else:
            logger.error("Error executing nvidia-smi")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

def normalize_output_for_prompt_baseline(text):
    if "yes" in text.lower() and "no" not in text.lower():
        return "Yes"
    elif "no" in text.lower() and "yes" not in text.lower():
        return "No"
    else:
        return "n/a"

def check_proposed_change(original_feature, proposed_feature):

    # Dictionary mapping desired features to a set of reasonable original features

    if proposed_feature not in desired_to_originals:
        # unreasonable_mappings.append((original_feature, proposed_feature))
        return False
    elif original_feature not in desired_to_originals[proposed_feature]:
        return False

    return True

def best_match_by_dict(original_feature, proposed_feature):
    if not check_proposed_change(original_feature, proposed_feature):
        for key, value in desired_to_originals.items():
            if original_feature in value:
                return key
    return None



def normalize_feature(feature):
    original_feature = feature
    feature = feature.strip()
    feature = re.sub(r'[^a-zA-Z0-9_]', '_', feature).lower()
    feature = re.sub(r'_+', '_', feature)
    return feature, original_feature

def find_closest_match(feature, desired_features):
    best_match = None
    best_ratio = 0

    for desired_feature in desired_features:
        ratio = SequenceMatcher(None, feature, desired_feature).ratio()
        if ratio > best_ratio:
            best_match = desired_feature
            best_ratio = ratio

    return best_match

def normalize_features(extracted_features, desired_features):
    normalized_features = []
    changes = {}
    unmatched_features = []
    total_features = len(extracted_features)
    converted_features = 0

    for feature in extracted_features:
        normalized_feature, original_feature = normalize_feature(feature)
        if normalized_feature in desired_features:
            normalized_features.append(normalized_feature)
            converted_features += 1
            if normalized_feature != original_feature:
                changes[original_feature] = normalized_feature
        else:
            closest_match = find_closest_match(normalized_feature, desired_features)
            if closest_match:
                normalized_features.append(closest_match)
                converted_features += 1
                changes[original_feature] = closest_match
            else:
                unmatched_features.append(original_feature)

    return normalized_features, changes, unmatched_features, total_features, converted_features
def get_highlight_data():
    extracted_feature_keys = set()

    # find the annotation schema in the paper and appendix
    with open("./data/extracted_features.jsonl", "r") as f_in:
        buf = [json.loads(line) for line in f_in]
        extracted_data = dict()
        for item in buf:
            _id = item["id"]
            # extracted_data[_id] = item
            valid_item = dict()
            for key in item:
                if item[key] is not None:
                    if isinstance(item[key], str):
                        if item[key].lower().strip() not in ["none", "", "null"]:
                            valid_item[key] = item[key]
                    else:
                        valid_item[key] = item[key]
                else:
                    continue
            extracted_feature_keys.update(valid_item.keys())
            extracted_data[_id] = valid_item
    # find all_features data
    with open("./data/ai_realtor_listing_data.json", "r") as f_in:
        buf = [json.loads(line) for line in f_in]
        all_features = dict()
        for item in buf:
            _id = item["id"]
            item.pop("description")
            all_features[_id] = item
    # randomly pickup one data in all_features to output the keys
    random_key = list(all_features.keys())[0]
    print("Keys in all_features:", len(all_features[random_key].keys()))
    # print(all_features[random_key].keys())
    print("Keys in extracted_features:", len(extracted_feature_keys))
    # print(extracted_feature_keys)
    # find the intersection of keys
    common_keys = extracted_feature_keys.intersection(all_features[random_key].keys())
    print("Common keys:", len(common_keys))
    normalized_features, changes, unmatched_features, total_features, converted_features = normalize_features(
        extracted_feature_keys, DESIRED_FEATURE_NAMES)
    print("Normalized Features:")
    print(normalized_features)

    print("\nChanges:")
    items = list(changes.items())
    for original_feature, normalized_feature in items:
        # print(f"{original_feature} -> {normalized_feature}")
        if check_proposed_change(original_feature, normalized_feature):
            print(f"{original_feature} -> {normalized_feature}")
        else:
            final_chance = best_match_by_dict(original_feature, normalized_feature)
            if final_chance is None:
                converted_features -= 1
                unmatched_features.append((original_feature, normalized_feature))
            else:
                changes[original_feature] = final_chance
                print(f"{original_feature} -> {final_chance}")

    print("\nConversion Statistics:")
    print(f"Total Features: {total_features}")
    print(f"Converted Features: {converted_features}")
    print(f"Unconverted Features: {len(unmatched_features)}")

    print("\nUnmatched Features:")
    print(unmatched_features)

    # use the changes to normalize features in extracted data
    for _id in extracted_data:
        keys = set(extracted_data[_id].keys())
        for key in keys:
            if key in changes:
                if changes[key] not in keys:
                    extracted_data[_id][changes[key]] = extracted_data[_id].pop(key)
                else:
                    extracted_data[_id].pop(key)

    return all_features, extracted_data
