from collections import Counter

import torch
from accelerate.commands.config.update import description

from utils import load_schema, load_hf_dataset

if __name__ == '__main__':
    ckpt_file_name = "zillow_feature_embedding_claude_schema_v9_fix_input_SFR-Embedding-Mistral_top_2000.ckpt"
    extracted_data, counter, ids, processed_ids = torch.load(ckpt_file_name)
    schemas = load_schema("claude_output/claude_merged.json")
    all_output_features = list(set(schemas.keys()))
    all_output_features.sort()
    label_counter = Counter()
    original_hf_ds = load_hf_dataset()
    original_id2data = {x['id']: x for x in original_hf_ds}
    id2labels = {}
    keyword_match_rates = {"matched_correct": 0, "matched_incorrect": 0, "unmatched_correct": 0, "unmatched_incorrect": 0}
    for _id in extracted_data:
        id2labels[_id] = set()
        for key in extracted_data[_id]:
            if key in ["embeddings", "id"]:
                continue
            assert key in all_output_features, "Key {} not in all output features".format(key)
            label_counter[key] += 1
            id2labels[_id].add(key)
    label_distribution = {k: v / len(extracted_data) for k, v in label_counter.items()}
    threshold = 0.5
    label_naive_bayes = {k: v > threshold for k, v in label_distribution.items()}
    banned_classes = dict()
    for k, v in label_distribution.items():
        if v > 0.7 or v < 0.3:
            print(k, v)
            banned_classes[k] = v
    torch.save(banned_classes, "too_freq_low_freq_banned_classes_claude_schema_llama3_70b_extraction_top2000.pt")

    for _id in extracted_data:
        description = original_id2data[_id]['description']
        for key in schemas:
            if key in banned_classes:
                continue
            keywords = schemas[key]
            if any([kw in description for kw in keywords]):
                if key in id2labels[_id]:
                    keyword_match_rates["matched_correct"] += 1
                else:
                    keyword_match_rates["matched_incorrect"] += 1
            else:
                if key in id2labels[_id]:
                    keyword_match_rates["unmatched_incorrect"] += 1
                else:
                    keyword_match_rates["unmatched_correct"] += 1
    # baseline 1: naive bayes
    correct = 0
    for _id in id2labels:
        for label in schemas:
            if label in banned_classes:
                continue
            if label_naive_bayes[label] and label in id2labels[_id]:
                correct += 1
            if not label_naive_bayes[label] and label not in id2labels[_id]:
                correct += 1
    accuracy = correct / (len(extracted_data) * (len(schemas) - len(banned_classes)))
    print(f"Naive Bayes accuracy: {accuracy}")
    # baseline 2: keyword matching
    matched_correct = keyword_match_rates["matched_correct"]
    matched_incorrect = keyword_match_rates["matched_incorrect"]
    unmatched_correct = keyword_match_rates["unmatched_correct"]
    unmatched_incorrect = keyword_match_rates["unmatched_incorrect"]
    accuracy = (matched_correct + unmatched_correct) / (matched_correct + matched_incorrect + unmatched_correct + unmatched_incorrect)
    print(f"Keyword matching accuracy (Note this involves description): {accuracy}")


