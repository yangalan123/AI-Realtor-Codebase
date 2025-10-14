import os
from collections import defaultdict, Counter
import torch
from utils import load_schema
if __name__ == '__main__':
    output_root_dir = "checkpoints_v10_sfr_mistral_top_2000_fix_input/evaluation"
    # train_dataset, test_dataset, dataset, all_output_features = torch.load(os.path.join(output_root_dir, "all_dataset.pt"))
    schemas = load_schema("claude_output/claude_merged_after_baseline_fix_0830.json")
    all_output_features = list(set(schemas.keys()))
    all_output_features.sort()
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



