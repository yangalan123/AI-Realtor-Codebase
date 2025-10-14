import os.path
from collections import Counter

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from utils import get_highlight_data, normalize_output_for_prompt_baseline
from const import DESIRED_FEATURE_NAMES, POST_PROCESSING_BANNED_LOW_FREQUENCY_FEATURES

if __name__ == '__main__':
    all_features_but_desc, extracted_data = get_highlight_data()
    ids = list(extracted_data.keys())
    # model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    output_set = Counter()
    all_outputs = []
    all_sample_ids = []
    all_prompts = []
    all_original_outputs = []
    for batch_id in range(10):
    # for batch_id in range(10):
        checkpoint_filename = "prompting_baseline_outputs/gpt-4o/highlight_model_prompting_gpt4_output_{}.pt".format(batch_id)
        if not os.path.exists(checkpoint_filename):
            print("File not found: {}".format(checkpoint_filename))
            break
        responses, test_sample_ids, prompts = torch.load(checkpoint_filename)
        for response in responses:
            output = response.choices[0].message.content
            # token_ids = output.token_ids
            # tokens = tokenizer.decode(token_ids[2] if len(token_ids) > 2 else token_ids[:1], skip_special_tokens=True)
            # output_set[tokens] += 1
            # tokens_complete = output.text
            # all_outputs.append(
            #     "The following is a response to a question. "
            #     "The possible answer candidates are only ['Yes', 'No', 'n/a'] (regardless of typo, grammar and lowercase). "
            #     "Can you read the response and tell me which answer it belongs to? \n\nResponse:" + tokens_complete + "\n\nAnswer:")
            all_outputs.append(output)
        all_sample_ids.extend(test_sample_ids)
        all_prompts.extend(prompts)
        # all_original_outputs.extend([response.outputs[0].text for response in responses])
            # if len(all_samples) < 20:
            #     all_samples.append(tokens_complete.strip())
    # print(output_set)
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
    for id_i, (_id, key) in enumerate(all_sample_ids):
        if _id not in id_wise_acc:
            id_wise_acc[_id] = []
        # original text as a judge
        # Feature - wise accuracy: 0.6357058823529412
        # ID - wise accuracy: 0.0
        # original_output = all_original_outputs[id_i]
        original_output = all_outputs[id_i]
        normalized_output = normalize_output_for_prompt_baseline(original_output)
        # Feature - wise accuracy: 0.605764705882353
        # ID - wise accuracy: 0.0
        # text = texts[id_i]
        # normalized_output = normalize_output_for_prompt_baseline(text)
        if key in POST_PROCESSING_BANNED_LOW_FREQUENCY_FEATURES:
            continue
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
    print("Feature-wise accuracy: {}".format(feature_wise_acc[0] / feature_wise_acc[1]))
    em_acc = 0
    count = 1e-6
    for _id in id_wise_acc:
        if len(id_wise_acc[_id]) == len(DESIRED_FEATURE_NAMES):
            count += 1
            if all(id_wise_acc[_id]):
                em_acc += 1
    print("ID-wise accuracy: {} ({}, {})".format(em_acc / count, em_acc, count))
