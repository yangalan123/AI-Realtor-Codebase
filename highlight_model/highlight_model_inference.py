import copy
import json
import os
from utils import load_schema, judge_empty_value, load_hf_dataset
import torch
from model import SimpleClassifier
from model import collate_fn_mlp as collate_fn
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
from vllm import LLM
from datasets import load_dataset

if __name__ == '__main__':
    #infer_data_filename = "listings_chi_sft_generated.json"
    #with open(infer_data_filename, "r", encoding='utf-8') as f_in:
        #infer_data = json.load(f_in)
    infer_data_filename = "chicago_dataset.json"
    with open(infer_data_filename, "r", encoding='utf-8') as f_in:
        infer_data = []
        for line in f_in:
            infer_data.append(json.loads(line))
    print("Load {} data to infer highlights from raw features".format(len(infer_data)))
    embed_model_name = "Salesforce/SFR-Embedding-Mistral"
    embedding_ckpt = infer_data_filename.replace(".json", "_embeddings.ckpt")
    banned_keys = ["description", 'url', "jpeg_urls", "id"]
    hf_dataset = load_dataset("[hf-path]")['train']
    raw_feature_list = set()
    for _datum in hf_dataset:
        for key in _datum:
            if key in banned_keys:
                continue
            if judge_empty_value(_datum[key]):
                raw_feature_list.add(key)
        break
    infer_ids = set([x['id'] for x in infer_data])
    hf_dataset = hf_dataset.filter(lambda x: x['id'] in infer_ids)
    id2original_data = {x['id']: x for x in hf_dataset}
    print(len(id2original_data))

    if not os.path.exists(embedding_ckpt):
        # embed_model = SentenceTransformer(embed_model_name)
        # pool = embed_model.start_multi_process_pool()
        embed_model = LLM(embed_model_name, enforce_eager=True)
        embed_profile = dict()
        for _datum in tqdm(infer_data):
            _id = _datum['id']
            _original_datum = id2original_data[_id]
            inputs = []
            classes = []
            for key in _original_datum:
                if key in banned_keys:
                    continue
                if judge_empty_value(_original_datum[key]):
                    inputs.append("The feature '{}' is '{}'.".format(key, _original_datum[key]))
                    classes.append(key)
            outputs = embed_model.encode(inputs)
            embeddings = []
            for output in outputs:
                embeddings.append(output.outputs.embedding)
            # embeddings = embed_model.encode_multi_process(inputs, pool)
            embed_profile[_id] = {"embeddings": embeddings, "classes": classes}
        torch.save(embed_profile, embedding_ckpt)
        print("Embeddings saved to {}".format(embedding_ckpt))
        print("You have to re-run the program to clear CUDA cache and memory for later use.")
        exit()
    else:
        embed_profile = torch.load(embedding_ckpt)

    # schemas = load_schema("claude_output/claude_merged.json")
    schemas = load_schema("claude_output/claude_merged_after_baseline_fix_0830.json")
    # train_ds, test_ds, ds, all_output_features = torch.load("checkpoints_v9_sfr_mistral_top_2000_fix_input/all_dataset.pt")
    train_ds, test_ds, ds, all_output_features = torch.load("checkpoints_v10_sfr_mistral_top_2000_fix_input/all_dataset.pt")
    dim = 4096
    model = SimpleClassifier(dim, len(all_output_features)).cuda()
    # state_dict = torch.load("checkpoints_v9_sfr_mistral_top_2000_fix_input/feature_classifier_best_mlp_mean.pt", weights_only=True)
    state_dict = torch.load("checkpoints_v10_sfr_mistral_top_2000_fix_input/feature_classifier_best_mlp_mean.pt", weights_only=True)
    for key in list(state_dict.keys()):
        if key.startswith("module."):
            state_dict[key[7:]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    output_data = []
    debug_flag = True
    for datum in tqdm(infer_data):
        _id = datum['id']
        assert _id in embed_profile, "ID {} not in embed profile".format(_id)
        all_embeddings = embed_profile[_id]["embeddings"]
        all_classes = embed_profile[_id]["classes"]
        assert len(all_embeddings) == len(all_classes), "Length mismatch"
        if debug_flag:
            print("added features:")
            print(set(all_classes) - raw_feature_list)
            print("removed features:")
            print(raw_feature_list - set(all_classes))
            removed_features = raw_feature_list - set(all_classes)
            removed_and_non_empty_features = [x for x in removed_features if not judge_empty_value(id2original_data[_id][x])]
            print("removed and non-empty features:")
            print(removed_and_non_empty_features)
            debug_flag = False
        embeddings = np.array([x for x, y in zip(all_embeddings, all_classes) if y in raw_feature_list])
        inputs = torch.tensor(embeddings.mean(axis=0)).unsqueeze(0).float().cuda()
        outputs = model(inputs)
        outputs = outputs.squeeze(0).detach().cpu().numpy()
        predicted_features = {all_output_features[i]: float(outputs[i]) for i in range(len(all_output_features))}
        # predicted_features = [all_output_features[i] for i in range(len(all_output_features)) if outputs[i] > 0.5]
        new_datum = copy.deepcopy(datum)
        new_datum["predicted_features"] = predicted_features
        output_data.append(new_datum)
    with open(infer_data_filename.replace(".json", "_output_fixed_v10.json"), "w", encoding='utf-8') as f_out:
        json.dump(output_data, f_out, indent=4)
        # datum["outputs"] = {all_output_features[i]: outputs[i] for i in range(len(all_output_features))}
