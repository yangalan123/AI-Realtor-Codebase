import torch
import os
from loguru import logger
import numpy as np
from const import DESIRED_FEATURE_NAMES, desired_to_originals, POST_PROCESSING_BANNED_LOW_FREQUENCY_FEATURES
from datasets import Dataset
from utils import load_schema
import copy
from accelerate import Accelerator
from tqdm import tqdm
# import datasets
# datasets.config.DEFAULT_MAX_BATCH_SIZE = 100
from utils import get_gpu_memory_and_usage_rate, judge_empty_value, get_highlight_data_claude_schema
import evaluate
from model import XFMRClassifier
import argparse
NUM_HEADS=8
# define a simple classification torch model, takes in embeddings and output labels
# each label classification is treated as a binary classification problem


def collate_fn(examples, vector_dictionary):
    inputs = []
    labels = []
    masks = []
    for example in examples:
        # inputs.append(torch.tensor(example["input"]))
        # overflow problem would occur, delay this to the collate_fn
        inputs.append(torch.tensor(np.array([vector_dictionary[feature] for feature in example["input"]])))
        labels.append(torch.tensor(example["labels"]))
        masks.append(torch.ones(len(example['input']), dtype=torch.float32))

    # inputs = torch.stack(inputs)
    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    masks = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True)
    batch_size, seq_len = masks.size()
    masks_unsqueeze = masks.unsqueeze(2)
    masks = masks_unsqueeze @ masks_unsqueeze.transpose(1, 2)
    masks = masks.unsqueeze(1).expand(-1, NUM_HEADS, -1, -1)
    masks = masks.reshape(batch_size * NUM_HEADS, seq_len, seq_len)

    labels = torch.stack(labels)
    return {"input": inputs, "mask": masks, "labels": labels}

if __name__ == '__main__':
    logger.add("feature_classifier_xfmr.log", rotation="10 MB")
    parser = argparse.ArgumentParser(description='EmbeddingArgsParser.')
    # parser.add_argument("--schema_extraction_filename", type=str, default="highlight_model_Mixtral-8x7B-Instruct-v0.1_extracted_features_all_data_v8_claude_schema_after_human_annotation/checkpoint_merged.pt", help="schema extraction filename")
    parser.add_argument("--schema_extraction_filename", type=str, default="highlight_model_Meta-Llama-3.1-70B-Instruct_extracted_features_all_data_v9_claude_schema_after_human_annotation_top_2000/checkpoint_merged.pt", help="schema extraction filename")
    parser.add_argument("--rating_by_score", action="store_true", help="rating by score")
    parser.add_argument("--top_k", type=int, default=2000, help="top k")
    parser.add_argument("--filter_by_city", type=str, default="", help="filter by city")
    args = parser.parse_args()

    # all_features, extracted_data = get_highlight_data()
    # all_features, extracted_data = get_highlight_data_claude_schema(
    #     schema_extraction_filename="highlight_model_Mixtral-8x7B-Instruct-v0.1_extracted_features_all_data_v8_claude_schema_after_human_annotation/checkpoint_merged.pt",
    #     rating_by_score=True, top_k=500, filter_by_city="chicago")
    all_features, _ = get_highlight_data_claude_schema(
        schema_extraction_filename=args.schema_extraction_filename,
        rating_by_score=args.rating_by_score, top_k=args.top_k, filter_by_city=args.filter_by_city)
    accelerator = Accelerator()
    # ckpt_file_name = "zillow_feature_embedding.ckpt"
    # ckpt_file_name = "zillow_feature_embedding_claude_schema_v9_SFR-Embedding-Mistral_top_2000.ckpt"
    ckpt_file_name = "zillow_feature_embedding_claude_schema_v10_fix_input_schema_merge_SFR-Embedding-Mistral_top_2000.ckpt"
    # output_root_dir = "checkpoints"
    # output_root_dir = "checkpoints_v9_sfr_mistral_top_2000_xfmr"
    output_root_dir = "checkpoints_v10_sfr_mistral_top_2000_fix_input_xfmr"
    os.makedirs(output_root_dir, exist_ok=True)
    extracted_data, counter, ids, processed_ids = torch.load(ckpt_file_name)
    # all_features, _ = get_highlight_data()
    dataset = {"input": [], "labels": []}
    dim = None
    seed = 42
    # schemas = load_schema("claude_output/claude_merged.json")
    schemas = load_schema("claude_output/claude_merged_after_baseline_fix_0830.json")
    all_output_features = list(set(schemas.keys()))
    all_output_features.sort()
    unused_features = copy.deepcopy(all_output_features)
    # unused_features = set(DESIRED_FEATURE_NAMES)
    # output_all_features = list(set(DESIRED_FEATURE_NAMES) - POST_PROCESSING_BANNED_LOW_FREQUENCY_FEATURES)
    vector_dictionary = dict()
    for _id in extracted_data:
        extracted_keys = set()
        if "embeddings" not in extracted_data[_id]:
            continue
        for key in extracted_data[_id]:
            if key in ["embeddings", "id"]:
                continue
            assert key in all_output_features, "Key {} not in all output features".format(key)
            if judge_empty_value(extracted_data[_id][key]):
                extracted_keys.add(key)
                unused_features.discard(key)
        # for key in extracted_data[_id]:
        #     if key in DESIRED_FEATURE_NAMES and extracted_data[_id][key] is not None:
        #         extracted_keys.add(key)
        #         unused_features.discard(key)
        input_feature_list = []
        for key in all_features[_id]:
            if all_features[_id][key] is not None:
                input_feature_list.append("The feature '{}' is '{}'.".format(key, all_features[_id][key]))
        assert len(input_feature_list) == len(extracted_data[_id]['embeddings']), "The feature number is not the same. ID: {}, All Features: {}, Extracted Features: {}".format(_id, len(input_feature_list), len(extracted_data[_id]['embeddings']))
        for feature_i, feature in enumerate(input_feature_list):
            if feature not in vector_dictionary:
                vector_dictionary[feature] = extracted_data[_id]['embeddings'][feature_i]
            else:
                # just ignore it
                pass
                # assert np.linalg.norm(vector_dictionary[feature] - extracted_data[_id]['embeddings'][feature_i]) < 1e-6, "The feature embedding is not the same. ID: {}, Feature: {}".format(_id, feature)

        # labels = np.zeros(len(DESIRED_FEATURE_NAMES))
        # labels = np.zeros(len(output_all_features))
        labels = np.zeros(len(all_output_features))
        # for i, feature in enumerate(output_all_features):
        for i, feature in enumerate(all_output_features):
            if feature in extracted_keys:
                labels[i] = 1
        embeddings = extracted_data[_id]['embeddings']
        _data = {
            # "input": extracted_data[_id]['embeddings'],
            # overflow problem would occur, delay this to the collate_fn
            "input": input_feature_list,
            "labels": labels
        }
        if dim is None:
            dim = len(embeddings[0])
        else:
            assert dim == len(embeddings[0])
        dataset["input"].append(_data["input"])
        dataset["labels"].append(_data["labels"])
    logger.debug("Dataset size: {}, Dim: {}, Never-Used Feature Num: {}, All Output Dim: {}".format(len(dataset["input"]), dim, len(unused_features), len(DESIRED_FEATURE_NAMES)))
    dataset = Dataset.from_dict(dataset)
    split_dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=seed)
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]
    _instantiated_collate_fn = lambda examples: collate_fn(examples, vector_dictionary)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, collate_fn=_instantiated_collate_fn, batch_size=4)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, collate_fn=_instantiated_collate_fn, batch_size=32)
    # model = XFMRClassifier(dim, len(output_all_features))
    model = XFMRClassifier(dim, len(all_output_features))
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    criterion = torch.nn.BCELoss()
    model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(model, optimizer, train_dataloader, test_dataloader)
    device = accelerator.device
    best_em = 0
    best_accuracy = 0
    for epoch in range(100):
        model.train()
        for data in tqdm(train_dataloader, desc="Training at Epoch {}".format(epoch)):
            input_data = data["input"].to(device)
            mask = data['mask'].to(device)
            labels = data["labels"].to(device)
            optimizer.zero_grad()
            outputs = model(input_data, mask)
            loss = criterion(outputs, labels)
            # loss.backward()
            accelerator.backward(loss)
            optimizer.step()
        # total = 0
        # correct = 0
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                for eval_name, eval_dataloader in zip(["train", "test"], [train_dataloader, test_dataloader]):
                    acc_metric = evaluate.combine(['accuracy', 'f1', 'precision', 'recall'])
                    # we reuse accuracy metric for em, as our output is not string-string exact match so we cannot use original EM
                    em_metric = evaluate.load("accuracy")
                    # micro_acc = [0, 0]
                    # macro_acc = [0, 0]
                    predictions = []
                    references = []
                    for data in tqdm(eval_dataloader, desc="Testing at Epoch {}".format(epoch)):
                        input_data = data["input"].to(device)
                        mask = data['mask'].to(device)
                        labels = data["labels"].to(device)
                        outputs = model(input_data, mask)
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
                                accelerator.save(model.state_dict(), os.path.join(output_root_dir, "feature_classifier_best_xfmr.pt"))
                                logger.debug("Best EM/acc updated: {} {}, model saved".format(best_em, best_accuracy))
                        accelerator.save([predictions, references], os.path.join(output_root_dir, f"feature_classifier_best_predictions_xfmr_{eval_name}.pt"))
                        logger.debug("Epoch: {} ({}), EM: {}, Accuracy: {}".format(epoch, eval_name, _em, _acc))
                # get_gpu_memory_and_usage_rate(logger)
    # accelerator.save(model.state_dict(), "feature_classifier.pt")



