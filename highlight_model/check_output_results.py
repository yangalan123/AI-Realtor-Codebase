import os.path
import numpy as np
from collections import Counter

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from utils import get_highlight_data_claude_schema, normalize_output_for_prompt_baseline, load_schema
# from const import DESIRED_FEATURE_NAMES, POST_PROCESSING_BANNED_LOW_FREQUENCY_FEATURES
import argparse

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
    parser.add_argument("--rating_by_score", action="store_false", help="rating by score")
    parser.add_argument("--top_k", type=int, default=2000, help="top k")
    parser.add_argument("--filter_by_city", type=str, default="", help="filter by city")
    args = parser.parse_args()
    all_features, extracted_data = get_highlight_data_claude_schema(
        schema_extraction_filename=args.schema_extraction_filename,
        rating_by_score=args.rating_by_score, top_k=args.top_k, filter_by_city=args.filter_by_city)
    ids = list(extracted_data.keys())
    #model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    schemas = load_schema("claude_output/claude_merged.json")
    all_output_features = set(schemas.keys())
    # model_name = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    model_name = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    output_set = Counter()
    stop_reasons = Counter()
    finish_reasons = Counter()
    all_outputs = []
    all_sample_ids = []
    all_prompts = []
    all_original_outputs = []
    all_prompt_token_ids_length = []
    for batch_id in range(10):
        #checkpoint_filename = "prompting_baseline_outputs/Mixtral-8x7B-Instruct-v0.1_5-shot/highlight_model_prompting_output_{}.pt".format(batch_id)
        #checkpoint_filename = "prompting_baseline_outputs_claude_schema/Meta-Llama-3.1-70B-Instruct_0-shot/highlight_model_prompting_output_{}.pt".format(batch_id)
        checkpoint_filename = "prompting_baseline_outputs_claude_schema_fix_prompt_0827/Meta-Llama-3.1-70B-Instruct_5-shot/highlight_model_prompting_output_{}.pt".format(batch_id)
        if not os.path.exists(checkpoint_filename):
            print("File not found: {}".format(checkpoint_filename))
            break
        responses, test_sample_ids, prompts = torch.load(checkpoint_filename)
        for response in responses:
            output = response.outputs[0]
            token_ids = output.token_ids
            #tokens = tokenizer.decode(token_ids[2] if len(token_ids) > 2 else token_ids[:1], skip_special_tokens=True)
            tokens = output.text
            output_set[tokens] += 1
            all_prompt_token_ids_length.append(len(response.prompt_token_ids))
            if len(tokens) == 0:
                stop_reasons[output.stop_reason] += 1
                finish_reasons[output.finish_reason] += 1
            tokens_complete = output.text
            all_outputs.append(
                "The following is a response to a question. "
                "The possible answer candidates are only ['Yes', 'No', 'n/a'] (regardless of typo, grammar and lowercase). "
                "Can you read the response and tell me which answer it belongs to? \n\nResponse:" + tokens_complete + "\n\nAnswer:")
        all_sample_ids.extend(test_sample_ids)
        all_prompts.extend(prompts)
        all_original_outputs.extend([response.outputs[0].text for response in responses])
            # if len(all_samples) < 20:
            #     all_samples.append(tokens_complete.strip())
    print(output_set.most_common(20))
    print("stop_reasons: {}".format(stop_reasons))
    print("finish_reasons: {}".format(finish_reasons))
    print("Prompt token ids length stat: mean={}, std={}, max={}, min={}".format(np.mean(all_prompt_token_ids_length),
                                                                                 np.std(all_prompt_token_ids_length),
                                                                                 np.max(all_prompt_token_ids_length),
                                                                                 np.min(all_prompt_token_ids_length)))
    # print(all_samples)
    # process_filename = "highlight_model_prompting_output_all.pt"
    # if not os.path.exists(process_filename):
    #     llm = LLM(model=model_name, tensor_parallel_size=4, gpu_memory_utilization=0.8, max_logprobs=10, seed=42)
    #     sampling_params = SamplingParams(n=1, max_tokens=10, logprobs=10)
    #     responses = llm.generate(all_outputs, sampling_params, use_tqdm=True)
    #     torch.save(responses, process_filename)
    # else:
    #     responses = torch.load(process_filename)
    # texts = [response.outputs[0].text for response in responses]
    feature_wise_acc = [0, 0]
    id_wise_acc = dict()
    normalized_outcomes = Counter()
    na_original_outcomes = Counter()
    for id_i, (_id, key) in enumerate(all_sample_ids):
        if _id not in id_wise_acc:
            id_wise_acc[_id] = []
        # original text as a judge
        # Feature - wise accuracy: 0.6357058823529412
        # ID - wise accuracy: 0.0
        original_output = all_original_outputs[id_i]
        normalized_output = normalize_output_for_prompt_baseline(original_output)
        normalized_outcomes[normalized_output] += 1
        if normalized_output == "n/a":
            na_original_outcomes[original_output] += 1
            continue
        # Feature - wise accuracy: 0.605764705882353
        # ID - wise accuracy: 0.0
        # text = texts[id_i]
        # normalized_output = normalize_output_for_prompt_baseline(text)
        # if key in POST_PROCESSING_BANNED_LOW_FREQUENCY_FEATURES:
        #     continue
        if key in extracted_data[_id]:
            if normalized_output == "Yes":
                feature_wise_acc[0] += 1
                id_wise_acc[_id].append(1)
            else:
                id_wise_acc[_id].append(0)
        else:
            if normalized_output == "No":
                feature_wise_acc[0] += 1
                id_wise_acc[_id].append(1)
            else:
                id_wise_acc[_id].append(0)
        feature_wise_acc[1] += 1
    print(normalized_outcomes)
    print(na_original_outcomes.most_common(20))
    print("Feature-wise accuracy: {}".format(feature_wise_acc[0] / feature_wise_acc[1]))
    em_acc = 0
    count = 1e-6
    for _id in id_wise_acc:
        if len(id_wise_acc[_id]) == len(all_output_features):
            count += 1
            if all(id_wise_acc[_id]):
                em_acc += 1
    print("ID-wise accuracy: {} ({}, {})".format(em_acc / count, em_acc, count))
