import copy

from model import SimpleClassifier
from model import collate_fn_mlp as collate_fn
import torch
import accelerate
import os
from loguru import logger
import numpy as np
from tqdm import tqdm
from utils import get_highlight_data



if __name__ == '__main__':
    accelerator = accelerate.Accelerator()
    output_root_dir = "checkpoints"
    train_dataset, test_dataset, dataset, all_output_features = torch.load(os.path.join(output_root_dir, "all_dataset.pt"))
    model = SimpleClassifier(len(train_dataset["input"][0]), len(train_dataset["labels"][0]))
    model.load_state_dict(torch.load(os.path.join(output_root_dir, "feature_classifier_best_mlp_mean.pt")))
    full_dataloader = torch.utils.data.DataLoader(dataset, collate_fn=collate_fn, batch_size=32)
    model, full_dataloader = accelerator.prepare(model, full_dataloader)
    device = accelerator.device
    model.eval()
    all_outputs = []
    with torch.no_grad():
        for data in full_dataloader:
            input_data = data["input"]
            labels = data["labels"]
            ids = data['id']
            input_data = input_data.to(device)
            labels = labels.to(device)
            outputs = model(input_data)
            outputs_cpu = outputs.cpu().numpy()
            assert len(ids) == len(outputs_cpu), "The ID size is not the same as the output size. ID Size: {}, Output Size: {}".format(len(ids), len(outputs_cpu))
            for _id, _output in zip(ids, outputs_cpu):
                _new_output = {"id": _id}
                for feature_i, feature in enumerate(all_output_features):
                    _new_output[feature] = float(_output[feature_i])
                all_outputs.append(_new_output)
    if accelerator.is_main_process:
        # write all_outputs to a jsonl file
        import json
        output_file_name = os.path.join(output_root_dir, "all_outputs.jsonl")
        with open(output_file_name, "w") as f:
            for _output in all_outputs:
                f.write(json.dumps(_output) + "\n")
        with open(os.path.join(output_root_dir, "train_ids.json"), "w") as f:
            json.dump(train_dataset["id"], f)
        test_ids = test_dataset["id"]
        output_file_name = os.path.join(output_root_dir, "test_outputs.jsonl")
        all_features, extracted_data_origin = get_highlight_data()
        with open(output_file_name, "w") as f:
            for _output in all_outputs:
                if _output["id"] in test_ids:
                    _new_data = copy.deepcopy(all_features[_output["id"]])
                    assert "highlight_feature_model" not in _new_data
                    _highlight = copy.deepcopy(_output)
                    _highlight.pop("id")
                    _new_data["highlight_feature_model"] = _highlight
                    f.write(json.dumps(_new_data) + "\n")

        logger.debug("All outputs are saved.")



