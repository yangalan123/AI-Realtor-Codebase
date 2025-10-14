import os
from collections import defaultdict, Counter
import torch
from utils import load_schema
if __name__ == '__main__':
    output_root_dir = "checkpoints_v10_sfr_mistral_top_2000_fix_input_2layerModel_raw_feature_prediction_baseline"
    # train_dataset, test_dataset, dataset, all_output_features = torch.load(os.path.join(output_root_dir, "all_dataset.pt"))
    # schemas = load_schema("claude_output/claude_merged_after_baseline_fix_0830.json")
    # all_output_features = list(set(schemas.keys()))
    # all_output_features.sort()
    root_dir = "path/to/dir/highlight_extraction/highlight_model_Meta-Llama-3.1-70B-Instruct_extracted_features_all_data_predict_raw_feature_baseline_top_2000"
    ckpt_prompt_name = f"{root_dir}/prompts.pt"

    total_prompts, scores, schema_dict, id2range, all_output_features = torch.load(ckpt_prompt_name)
    for eval_set in ['train', "test"]:
        performance_metrics_classwise = defaultdict(Counter)
        [predictions, references] = torch.load(os.path.join(output_root_dir, f"feature_classifier_best_predictions_mlp_mean_{eval_set}.pt"))
        print(f"Eval Set: {eval_set}")
        assert len(predictions) == len(references), f"Length mismatch between predictions and references for {eval_set}"
        all_data_num = 0
        for outputs, labels in zip(predictions, references):
            num_data = outputs.shape[0]
            num_labels = outputs.shape[1]
            all_data_num += num_data
            assert num_labels == len(all_output_features), f"Length mismatch between number of output features and number of labels for {eval_set}"
            judge = (outputs == labels)
            for label_j in range(num_labels):
                label_name = all_output_features[label_j]
                performance_metrics_classwise[label_name]["correct"] += judge[:, label_j].sum().item()
                performance_metrics_classwise[label_name]["total"] += num_data
                performance_metrics_classwise[label_name]["Existing"] += labels[:, label_j].sum().item()
        performance_metrics_classwise_final = []
        for label_name in all_output_features:
            if label_name in ["view", "favorite_count", "scraped_at"]:
                continue
            correct = performance_metrics_classwise[label_name]["correct"]
            total = performance_metrics_classwise[label_name]["total"]
            accuracy = correct / total
            freq = performance_metrics_classwise[label_name]["Existing"] / all_data_num
            performance_metrics_classwise_final.append((label_name, accuracy, freq))
        performance_metrics_classwise_final.sort(key=lambda x: x[1], reverse=True)
        # print top-20 features
        for label_name, accuracy, frequency in performance_metrics_classwise_final[:20]:
            print(f"Label: {label_name}, Accuracy: {accuracy}, Frequency: {frequency}")
            # print(f"Label: {label_name}, Accuracy: {accuracy}")



