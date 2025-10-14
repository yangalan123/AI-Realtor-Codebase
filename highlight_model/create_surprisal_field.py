import copy
import json
from tqdm import tqdm

def load_json_or_jsonl_data(filepath):
    with open(filepath, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            return [json.loads(line) for line in f]

if __name__ == '__main__':
    retrieval_results_fn = "path/to/dir/rag_agents/rag_ckpt/retrieval_top_listings_chi_sft_generated.json_10_es.json"
    highlight_results_fn = "path/to/dir/highlight_extraction/chicago_dataset_output_fixed_v10.json"
    sourcefile_fn = "path/to/dir/highlight_extraction/listings_chi_sft_generated_output_fixed_v10.json"
    retrieval_results = load_json_or_jsonl_data(retrieval_results_fn)
    highlight_results = load_json_or_jsonl_data(highlight_results_fn)
    source_data = load_json_or_jsonl_data(sourcefile_fn)
    output_file = sourcefile_fn.replace(".json", "_with_retrieval_and_surprisal_top10.json")
    id_to_highlights = dict()
    for item in highlight_results:
        id_to_highlights[item["id"]] = item['predicted_features']
    id_to_neighbors = dict()
    for item in retrieval_results:
        id_to_neighbors[item["id"]] = item['retrieved_results']

    new_data_buffer = []
    for item in tqdm(source_data):
        _id = item['id']
        new_item = copy.deepcopy(item)
        new_item["retrieval_results"] = dict()
        new_item["retrieval_results"]["ids"] = id_to_neighbors[_id]
        new_item["retrieval_results"]["highlights"] = [id_to_highlights[x] for x in id_to_neighbors[_id]]
        current_highlight = id_to_highlights[_id]
        if len(id_to_neighbors[_id]) == 0:
            print("No neighbors for id: ", _id)
            new_item['surprisal_scores'] = copy.deepcopy(current_highlight)
        else:
            new_item['surprisal_scores'] = dict()
            for highlight in current_highlight:
                neighborhood_scores = [x[highlight] for x in new_item["retrieval_results"]["highlights"]]
                aggr_score = max(neighborhood_scores)
                if current_highlight[highlight] >= aggr_score:
                    new_item["surprisal_scores"][highlight] = current_highlight[highlight] - aggr_score
                else:
                    new_item["surprisal_scores"][highlight] = 0
        new_data_buffer.append(new_item)
    with open(output_file, "w") as f_out:
        for item in new_data_buffer:
            f_out.write(json.dumps(item) + "\n")

