import torch
import copy
import os
from loguru import logger
import numpy as np
# from const import DESIRED_FEATURE_NAMES, desired_to_originals, POST_PROCESSING_BANNED_LOW_FREQUENCY_FEATURES
from datasets import Dataset
from accelerate import Accelerator
from utils import get_gpu_memory_and_usage_rate, judge_empty_value, load_schema
import evaluate
from model import SimpleClassifier
from model import collate_fn_mlp as collate_fn

# define a simple classification torch model, takes in embeddings and output labels
# each label classification is treated as a binary classification problem
if __name__ == '__main__':
    logger.add("eval_highlight_model_2layer.log", rotation="10 MB")
    ckpt_file_name = "zillow_feature_embedding_claude_schema_v10_fix_input_schema_merge_SFR-Embedding-Mistral_top_2000.ckpt"
    #output_root_dir = "checkpoints_v9_sfr_mistral_top_2000"
    # output_root_dir = "checkpoints_v9_sfr_mistral_top_2000_fix_input"
    # output_root_dir = "checkpoints_v10_sfr_mistral_top_2000_fix_input/evaluation"
    output_root_dir = "checkpoints_v10_sfr_mistral_top_2000_fix_input_2layerModel/evaluation"
    # get parent directory
    output_root = os.path.dirname(output_root_dir)
    os.makedirs(output_root_dir, exist_ok=True)
    extracted_data, counter, ids, processed_ids = torch.load(ckpt_file_name)
    schemas = load_schema("claude_output/claude_merged_after_baseline_fix_0830.json")
    # train_ds, test_ds, ds, all_output_features = torch.load("checkpoints_v10_sfr_mistral_top_2000_fix_input/all_dataset.pt")
    train_ds, test_ds, ds, all_output_features = torch.load(f"{output_root}/all_dataset.pt")
    dim = 4096
    if "layer" in output_root_dir:
        model = SimpleClassifier(dim, 1024, len(all_output_features)).cuda()
    else:
        model = SimpleClassifier(dim, None, len(all_output_features)).cuda()
    # state_dict = torch.load("checkpoints_v9_sfr_mistral_top_2000_fix_input/feature_classifier_best_mlp_mean.pt", weights_only=True)
    state_dict = torch.load(f"{output_root}/feature_classifier_best_mlp_mean.pt", weights_only=True)
    for key in list(state_dict.keys()):
        if key.startswith("module."):
            state_dict[key[7:]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    train_dataloader = torch.utils.data.DataLoader(train_ds, collate_fn=collate_fn, batch_size=4)
    test_dataloader = torch.utils.data.DataLoader(test_ds, collate_fn=collate_fn, batch_size=32)
    model.eval()
    with torch.no_grad():
        for eval_name, dataloader in zip(["train", "test"], [train_dataloader, test_dataloader]):
            acc_metric = evaluate.combine(['accuracy', 'f1', 'precision', 'recall'])
            # we reuse accuracy metric for em, as our output is not string-string exact match so we cannot use original EM
            em_metric = evaluate.load("accuracy")
            predictions = []
            references = []
            for data in dataloader:
                input_data = data["input"].cuda()
                labels = data["labels"].cuda()
                outputs = model(input_data)
                outputs = (outputs > 0.5).float()
                judge = (outputs == labels).float()
                num_data = outputs.size(0)
                num_labels = outputs.size(1)
                sample_correct = judge.mean(dim=1)
                predictions.append(outputs.int().cpu().numpy())
                references.append(labels.int().cpu().numpy())
                # macro_acc[0] += (judge.sum(dim=1) == num_labels).float().sum().item()
                # macro_acc[1] += num_data
                # micro_acc[0] += judge.sum().item()
                # micro_acc[1] += num_data * num_labels
                acc_metric.add_batch(predictions=outputs.int().reshape(-1), references=labels.int().reshape(-1))
                em_metric.add_batch(predictions=(judge.sum(dim=1) == num_labels).int().cpu(),
                                    references=torch.ones(num_data).int().cpu())

            torch.save([predictions, references],
                             os.path.join(output_root_dir, f"feature_classifier_best_predictions_mlp_mean_{eval_name}.pt"))
            _em = em_metric.compute()
            _acc = acc_metric.compute()
            acc = _acc['accuracy']
            logger.debug("Eval Dataset: {}, EM: {}, Accuracy: {}".format(eval_name, _em, _acc))

