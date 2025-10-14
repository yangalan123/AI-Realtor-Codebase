import torch
import copy
import os
from loguru import logger
from collections import Counter
import numpy as np
# from const import DESIRED_FEATURE_NAMES, desired_to_originals, POST_PROCESSING_BANNED_LOW_FREQUENCY_FEATURES
from datasets import Dataset
from accelerate import Accelerator
from utils import get_gpu_memory_and_usage_rate, judge_empty_value, load_schema, load_hf_dataset
import evaluate
from model import SimpleClassifier
from model import collate_fn_mlp as collate_fn

# define a simple classification torch model, takes in embeddings and output labels
# each label classification is treated as a binary classification problem



if __name__ == '__main__':
    #logger.add("feature_classifier.log", rotation="10 MB")
    top_k = 10000
    flag_using_weighted_loss = True
    logger.add(f"feature_classifier_2layerMLPPooling_top{top_k}{'_weighted' if flag_using_weighted_loss else ''}.log", rotation="10 MB")
    accelerator = Accelerator()
    #ckpt_file_name = "zillow_feature_embedding.ckpt"
    #ckpt_file_name = "zillow_feature_embedding_claude_schema_v9_SFR-Embedding-Mistral_top_2000.ckpt"
    # ckpt_file_name = "zillow_feature_embedding_claude_schema_v9_fix_input_SFR-Embedding-Mistral_top_2000.ckpt"
    # ckpt_file_name = "zillow_feature_embedding_claude_schema_v10_fix_input_schema_merge_SFR-Embedding-Mistral_top_2000.ckpt"
    ckpt_file_name = f"zillow_feature_embedding_claude_schema_v10_fix_input_SFR-Embedding-Mistral_top_{top_k}.ckpt"
    #output_root_dir = "checkpoints_v9_sfr_mistral_top_2000"
    # output_root_dir = "checkpoints_v9_sfr_mistral_top_2000_fix_input"
    # output_root_dir = "checkpoints_v10_sfr_mistral_top_2000_fix_input_2layerModel"
    if flag_using_weighted_loss:
        output_root_dir = f"checkpoints_v10_sfr_mistral_top_{top_k}_fix_input_2layerModel_weighted"
    else:
        output_root_dir = f"checkpoints_v10_sfr_mistral_top_{top_k}_fix_input_2layerModel"
    os.makedirs(output_root_dir, exist_ok=True)
    extracted_data, counter, ids, processed_ids = torch.load(ckpt_file_name)
    all_features = load_hf_dataset(rating_by_score=True, top_k=top_k)
    id2score = {x['id']: x['score'] for x in all_features}
    schemas = load_schema("claude_output/claude_merged_after_baseline_fix_0830.json")
    all_output_features = list(set(schemas.keys()))
    all_output_features.sort()
    unused_features = set(copy.deepcopy(all_output_features))
    dataset = {"input": [], "labels": [], "id": [], "score": []}
    dim = None
    seed = 42
    # unused_features = set(DESIRED_FEATURE_NAMES) - set(POST_PROCESSING_BANNED_LOW_FREQUENCY_FEATURES)
    # all_output_features = list(set(DESIRED_FEATURE_NAMES) - set(POST_PROCESSING_BANNED_LOW_FREQUENCY_FEATURES))
    num_dims = []
    for _id in extracted_data:
        extracted_keys = set()
        for key in extracted_data[_id]:
            if key in ["embeddings", "id"]:
                continue
            assert key in all_output_features, "Key {} not in all output features".format(key)
            if judge_empty_value(extracted_data[_id][key]):
                extracted_keys.add(key)
                unused_features.discard(key)
        # labels = np.zeros(len(DESIRED_FEATURE_NAMES))
        labels = np.zeros(len(all_output_features))
        # for i, feature in enumerate(DESIRED_FEATURE_NAMES):
        for i, feature in enumerate(all_output_features):
            if feature in extracted_keys:
                labels[i] = 1
        if "embeddings" not in extracted_data[_id]:
            continue
        _data = {
            "input": extracted_data[_id]['embeddings'].mean(axis=0),
            "labels": labels,
            "id": _id,
            "score": id2score[_id]
        }
        num_dims.append(len(extracted_data[_id]['embeddings']))
        if dim is None:
            dim = len(_data["input"])
        else:
            assert dim == len(_data["input"])
        dataset["input"].append(_data["input"])
        dataset["labels"].append(_data["labels"])
        dataset["id"].append(_data["id"])
        dataset["score"].append(_data["score"])
    logger.debug("Dataset size: {}, Dim: {}, Never-Used Feature Num: {}, Effective Output Dim: {}".format(len(dataset["input"]), dim, len(unused_features), len(all_output_features) - len(unused_features)))
    logger.debug("num dims: {}".format(Counter(num_dims)))
    dataset = Dataset.from_dict(dataset)
    split_dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=seed)
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]
    accelerator.save([train_dataset, test_dataset, dataset, all_output_features], os.path.join(output_root_dir, "all_dataset.pt"))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, collate_fn=collate_fn, batch_size=4)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, collate_fn=collate_fn, batch_size=32)
    full_dataloader = torch.utils.data.DataLoader(dataset, collate_fn=collate_fn, batch_size=32)
    model = SimpleClassifier(dim, 1024, len(all_output_features))
    if os.path.exists(f"{output_root_dir}/feature_classifier_best_mlp_mean.pt"):
        state_dict = torch.load(f"{output_root_dir}/feature_classifier_best_mlp_mean.pt", weights_only=True)
        for key in list(state_dict.keys()):
            if key.startswith("module."):
                state_dict[key[7:]] = state_dict.pop(key)
        model.load_state_dict(state_dict)
        # copy the old file to .bak
        if accelerator.is_main_process:
            # os.rename(f"{output_root_dir}/feature_classifier_best_mlp_mean.pt", f"{output_root_dir}/feature_classifier_best_mlp_mean.pt.bak")
            # using copy! not rename!
            os.system(f"cp {output_root_dir}/feature_classifier_best_mlp_mean.pt {output_root_dir}/feature_classifier_best_mlp_mean.pt.bak")
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-6)
    criterion = torch.nn.BCELoss(reduction="none" if flag_using_weighted_loss else "mean")
    model, optimizer, train_dataloader, test_dataloader, full_dataloader = accelerator.prepare(model, optimizer, train_dataloader, test_dataloader, full_dataloader)
    device = accelerator.device
    best_em = 0
    best_accuracy = 0
    for epoch in range(1000):
        model.train()
        for data in train_dataloader:
            input_data = data["input"]
            labels = data["labels"]
            scores = data['score']
            optimizer.zero_grad()
            input_data = input_data.to(device)
            labels = labels.to(device)
            scores = scores.to(device)
            outputs = model(input_data)
            weights = torch.softmax(-scores, dim=0)
            if flag_using_weighted_loss:
                loss = criterion(outputs, labels)
                # broadcast the weights to the same shape as loss
                # labels: batch_size * len(all_output_features)
                # current weights: batch_size
                # we need to broadcast the weights to batch_size * len(all_output_features)
                weights = weights.unsqueeze(1).expand(-1, len(all_output_features))
                loss = (loss * weights).mean()
            else:
                loss = criterion(outputs, labels)
            # loss.backward()
            accelerator.backward(loss)
            optimizer.step()
        # total = 0
        # correct = 0
        if epoch % 5 == 0:
            model.eval()
            for eval_name, eval_dataloader in zip(['train', 'test'], [train_dataloader, test_dataloader]):
                acc_metric = evaluate.combine(['accuracy', 'f1', 'precision', 'recall'])
                # we reuse accuracy metric for em, as our output is not string-string exact match so we cannot use original EM
                em_metric = evaluate.load("accuracy")
                # micro_acc = [0, 0]
                # macro_acc = [0, 0]
                predictions = []
                references = []
                for data in eval_dataloader:
                    input_data = data["input"]
                    labels = data["labels"]
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
                    em_metric.add_batch(predictions=(judge.sum(dim=1) == num_labels).int().cpu(), references=torch.ones(num_data).int().cpu())
                    # assert outputs.size() == labels.size()
                    # print(labels.size(), outputs.size())
                    # total += labels.size(0)
                    # correct += (outputs == labels).sum().item()
                if accelerator.is_main_process:
                    _em = em_metric.compute()
                    _acc = acc_metric.compute()
                    acc = _acc['accuracy']
                    if eval_name == "test":
                        if _em['accuracy'] > best_em or (_em['accuracy'] == best_em and acc > best_accuracy):
                            best_em = _em['accuracy']
                            best_accuracy = acc
                            accelerator.save(model.state_dict(), os.path.join(output_root_dir, "feature_classifier_best_mlp_mean.pt"))
                            logger.debug("Best EM/acc updated: {} {}, model saved".format(best_em, best_accuracy))
                    accelerator.save([predictions, references], os.path.join(output_root_dir, f"feature_classifier_best_predictions_mlp_mean_{eval_name}.pt"))
                    logger.debug("[{}] Epoch: {}, EM: {}, Accuracy: {}".format(eval_name, epoch, _em, _acc))

                # get_gpu_memory_and_usage_rate(logger)
    # accelerator.save(model.state_dict(), "feature_classifier.pt")



