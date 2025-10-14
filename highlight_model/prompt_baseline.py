import argparse
import copy
import os
import random

import torch
from vllm import LLM, SamplingParams

#from const import DESIRED_FEATURE_NAMES
from utils import judge_empty_value, load_schema, get_highlight_data_claude_schema


def format_sample_data(_all_features, extract_data, with_answer=False):
    inputs = []
    for key in _all_features:
        if judge_empty_value(_all_features[key]) and key not in ['id', 'description', "embeddings", "embedding", "url", "jpeg_urls"]:
            inputs.append("The feature [{}] is [{}].".format(key, _all_features[key]))
    if with_answer:
        answer = []
        for key in extract_data:
            if extract_data[key] is not None and key not in ['id', 'description', "embeddings", "embedding", "url", "jpeg_urls"]:
                # answer.append("The feature [{}] is [{}].".format(key, extract_data[key]))
                answer.append("[{}]".format(key))
        prompt = "#ALL Features#\n\n{}\n\n#Highlighted Features#\n\n{}\n\n".format(" ".join(inputs), ",".join(answer))
        return prompt
    else:
        prompt = "#ALL Features#\n\n{}\n\n".format(" ".join(inputs))
        return prompt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RAGAgentArgsParser.')
    # parser.add_argument('--model', type=str, default="CohereForAI/c4ai-command-r-plus", help='model name')
    #parser.add_argument('--model', type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1", help='model name')
    parser.add_argument('--model', type=str, default="meta-llama/Meta-Llama-3.1-70B-Instruct", help='model name')
    # parser.add_argument('--model', type=str, default="mistralai/Mixtral-8x22B-Instruct-v0.1", help='model name')
    parser.add_argument("--max_tokens", type=int, default=64, help="max tokens for generation")
    parser.add_argument("--num_shots", type=int, default=5, help="number of shots for training")
    parser.add_argument("--schema_extraction_filename", type=str,
                        default="highlight_model_Meta-Llama-3.1-70B-Instruct_extracted_features_all_data_v9_claude_schema_after_human_annotation_top_2000/checkpoint_merged.pt",
                        help="schema extraction filename")
    parser.add_argument("--rating_by_score", action="store_true", help="rating by score")
    parser.add_argument("--top_k", type=int, default=2000, help="top k")
    parser.add_argument("--filter_by_city", type=str, default="", help="filter by city")
    args = parser.parse_args()
    # all_features_but_desc, extracted_data = get_highlight_data
    all_features, extracted_data = get_highlight_data_claude_schema(
        schema_extraction_filename=args.schema_extraction_filename,
        rating_by_score=args.rating_by_score, top_k=args.top_k, filter_by_city=args.filter_by_city)
    schemas = load_schema("claude_output/claude_merged.json")
    all_output_features = set(schemas.keys())
    unused_features = copy.deepcopy(all_output_features)
    ids = list(extracted_data.keys())
    sample_train_data = args.num_shots
    random.seed(42)
    # first search for existing files
    root_dir = "prompting_baseline_outputs_claude_schema_fix_prompt_0827"
    os.makedirs(root_dir, exist_ok=True)
    output_root_dir = "{}/{}_{}-shot".format(root_dir, os.path.basename(args.model), args.num_shots)
    os.makedirs(output_root_dir, exist_ok=True)
    train_sample_ids_filename = "{}/highlight_model_prompting_train_ids_{}shot_claude_schema.pt".format(root_dir, sample_train_data)
    if os.path.exists(train_sample_ids_filename):
        train_ids = torch.load(train_sample_ids_filename)
    else:
        train_ids = random.sample(ids, sample_train_data)
        torch.save(train_ids, train_sample_ids_filename)
    train_sample_prompts = [
        f"Your task is to identify the features highlighted in the following text given all the features. {'Here are some examples: ' if sample_train_data > 0 else ''}\n\n"]
    for id_i, _id in enumerate(train_ids):
        _all_features = all_features[_id]
        extracted_features = extracted_data[_id]
        prompt = format_sample_data(_all_features, extracted_features, with_answer=True)
        train_sample_prompts.append("---Sample {}---\n\n".format(id_i) + prompt)
    train_prompt = "".join(train_sample_prompts)
    print("Train Prompt:")
    print(train_prompt)

    prompts = []
    test_sample_ids = []
    ids_count = 0
    random.shuffle(ids)
    
    for _id in ids:
        if _id in train_ids:
            continue
        _all_features = all_features[_id]
        extracted_features = extracted_data[_id]
        prompt = format_sample_data(_all_features, extracted_features)
        #for key in DESIRED_FEATURE_NAMES:
        for key in all_output_features:
            # if _all_features[key] is not None:
            prompts.append(train_prompt +
                           "Now, you are given a new home listing with the following features: \n\n" +
                           prompt +
                           "Can you tell me whether the feature [{}] should be highlighted in the text? "
                           "You must only answer either “yes” or “no”.\n\nResponse (YES/NO):".format(
                               key))
            test_sample_ids.append((_id, key))
        ids_count += 1
        if ids_count >= 100:
            break
    print("Test Prompt Example:")
    print(prompts[0])

    llm = LLM(model=args.model, max_num_seqs=8, tensor_parallel_size=2, gpu_memory_utilization=0.98, max_logprobs=10, seed=42, max_model_len=16384)
    sampling_params = SamplingParams(n=1, max_tokens=args.max_tokens, logprobs=10)
    # split ids into batches, after each batch, save the results
    num_batches = 10
    batch_size = len(prompts) // num_batches
    batch_ids = list(range(num_batches))
    for i in range(num_batches):
        if os.path.exists(os.path.join(output_root_dir, "highlight_model_prompting_output_{}.pt".format(i))):
            batch_ids.remove(i)
    for i in batch_ids:
        start = i * batch_size
        end = (i + 1) * batch_size
        if i == num_batches - 1:
            end = len(prompts)
        responses = llm.generate(prompts[start:end], sampling_params, use_tqdm=True)
        torch.save([responses, test_sample_ids[start:end], prompts[start:end]],
                   os.path.join(output_root_dir, "highlight_model_prompting_output_{}.pt".format(i)))
    #
    #
    # responses = llm.generate(prompts, sampling_params, use_tqdm=True)
    # torch.save([responses, test_sample_ids, prompts], "highlight_model_prompting_output.pt")
