import os.path
import numpy as np
import copy

# from sentence_transformers import SentenceTransformer
from vllm import LLM
from tqdm import tqdm
import gc
import torch
from utils import get_highlight_data, get_highlight_data_claude_schema
import argparse



# Example usage
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EmbeddingArgsParser.')
    parser.add_argument('--model', type=str, default="Salesforce/SFR-Embedding-Mistral", help='embedding model name')
    # parser.add_argument("--schema_extraction_filename", type=str, default="highlight_model_Mixtral-8x7B-Instruct-v0.1_extracted_features_all_data_v8_claude_schema_after_human_annotation/checkpoint_merged.pt", help="schema extraction filename")
    parser.add_argument("--schema_extraction_filename", type=str, default="highlight_model_Meta-Llama-3.1-70B-Instruct_extracted_features_all_data_v10_claude_schema_after_human_annotation_top_10000/checkpoint_merged.pt", help="schema extraction filename")
    parser.add_argument("--rating_by_score", action="store_true", help="rating by score")
    parser.add_argument("--top_k", type=int, default=10000, help="top k")
    parser.add_argument("--filter_by_city", type=str, default="", help="filter by city")
    args = parser.parse_args()

    # all_features, extracted_data = get_highlight_data()
    # all_features, extracted_data = get_highlight_data_claude_schema(
    #     schema_extraction_filename="highlight_model_Mixtral-8x7B-Instruct-v0.1_extracted_features_all_data_v8_claude_schema_after_human_annotation/checkpoint_merged.pt",
    #     rating_by_score=True, top_k=500, filter_by_city="chicago")
    all_features, extracted_data = get_highlight_data_claude_schema(
        schema_extraction_filename=args.schema_extraction_filename,
        rating_by_score=args.rating_by_score, top_k=args.top_k, filter_by_city=args.filter_by_city)
    # model = SentenceTransformer("Salesforce/SFR-Embedding-Mistral")
    # model = SentenceTransformer(args.model)
    embed_model = LLM(args.model, tensor_parallel_size=4, enforce_eager=True)
    ids = list(extracted_data.keys())
    counter = 0
    processed_ids = set()
    # ckpt_file_name = "zillow_feature_embedding.ckpt"
    # ckpt_file_name = "zillow_feature_embedding_claude_schema.ckpt"
    # ckpt_file_name = f"zillow_feature_embedding_claude_schema_v8_{os.path.basename(args.model)}_{args.top_k}_{args.filter_by_city}.ckpt"
    # ckpt_file_name = f"zillow_feature_embedding_claude_schema_v9_{os.path.basename(args.model)}_{args.top_k}_{args.filter_by_city}.ckpt"
    #ckpt_file_name = "zillow_feature_embedding_claude_schema_v9_{}{}{}.ckpt".format(
    ckpt_file_name = "zillow_feature_embedding_claude_schema_v10_fix_input_{}{}{}.ckpt".format(
        os.path.basename(args.model),
        f"_top_{args.top_k}" if args.rating_by_score else "",
        f"_filterby_{args.filter_by_city}" if len(args.filter_by_city) > 0 else ""
    )
    if os.path.exists(ckpt_file_name):
        extracted_data, counter, ids, processed_ids = torch.load(ckpt_file_name)
        assert set(ids[:counter]) == processed_ids
        ids = ids[counter:]
    # pool = model.start_multi_process_pool()

    id2range = dict()
    inputs = []
    for _id in tqdm(ids, desc="Gather Prompts"):
        _all_features = all_features[_id]
        extracted_features = extracted_data[_id]
        assert _id not in id2range, "Duplicate id {}".format(_id)
        start_position = len(inputs)
        for key in _all_features:
            if _all_features[key] is not None:
                inputs.append("The feature '{}' is '{}'.".format(key, _all_features[key]))
        id2range[_id] = (start_position, len(inputs))

    outputs = embed_model.encode(inputs, use_tqdm=True)

    for _id in tqdm(ids, desc="Retrieve Embeddings"):
        # get the embeddings using the model
        # embeddings = embed_model.encode_multi_process(inputs, pool)
        embeddings = []
        start_position, end_position = id2range[_id]
        for output in outputs[start_position:end_position]:
            embeddings.append(output.outputs.embedding)
        # extracted_data[_id]["embeddings"] = copy.copy(embeddings)
        extracted_data[_id]["embeddings"] = np.array(embeddings)
        # del embeddings
        # del outputs
        # gc.collect()
        # torch.cuda.empty_cache()
        # counter += 1
        # processed_ids.add(_id)
        # if counter % 100 == 0:
    torch.save([extracted_data, counter, ids, processed_ids], ckpt_file_name)

    # model.stop_multi_process_pool(pool)
