import os.path
import copy

from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import gc
import torch
from utils import get_highlight_data, get_original_all_features_data



# Example usage
if __name__ == '__main__':

    all_features_original = get_original_all_features_data()
    all_features, _ = get_highlight_data()
    # SFT-Embedding-Mistral is already a bit outdated, you can use other models.
    model = SentenceTransformer("Salesforce/SFR-Embedding-Mistral")
    ids = [k for k in all_features_original.keys() if all_features_original[k]["city"].lower() == "chicago"]
    counter = 0
    processed_ids = set()
    id2features = dict()
    filtered_data = dict()
    # you can create your own embedding checkpoint filename. Do not panic, as here we are just checking whether ckpt file exists.
    ckpt_file_name = "./data/zillow_feature_embedding_with_description_chicago.ckpt"
    if os.path.exists(ckpt_file_name):
        filtered_data, counter, ids, processed_ids, id2features = torch.load(ckpt_file_name)
        assert set(ids[:counter]) == processed_ids
        ids = ids[counter:]
    pool = model.start_multi_process_pool()

    for _id in tqdm(ids, desc="Extracting embeddings"):
        _all_features = all_features[_id]
        # extracted_features = extracted_data[_id]
        if "description" not in all_features_original[_id] or all_features_original[_id]["description"] is None:
            continue
        inputs = []
        assert _id not in id2features
        id2features[_id] = []
        for key in _all_features:
            if _all_features[key] is not None:
                inputs.append("The feature '{}' is '{}'.".format(key, _all_features[key]))
                id2features[_id].append(key)
        description = all_features_original[_id]["description"]
        inputs.append(description)
        id2features[_id].append("description")
        filtered_data[_id] = copy.deepcopy(all_features_original[_id])

        # get the embeddings using the model
        embeddings = model.encode_multi_process(inputs, pool)
        filtered_data[_id]["embeddings"] = copy.copy(embeddings)
        # extracted_data[_id]["embeddings"] = copy.copy(embeddings)
        del embeddings
        gc.collect()
        torch.cuda.empty_cache()
        counter += 1
        processed_ids.add(_id)
        if counter % 100 == 0:
            torch.save([filtered_data, counter, ids, processed_ids, id2features], ckpt_file_name)

    model.stop_multi_process_pool(pool)
