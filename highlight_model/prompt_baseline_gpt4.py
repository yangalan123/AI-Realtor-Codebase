from utils import get_highlight_data
import time
import os
import torch
import random
# from vllm import LLM, SamplingParams
from openai import OpenAI
from const import DESIRED_FEATURE_NAMES
import argparse
from tqdm import tqdm

def format_sample_data(_all_features, extract_data, with_answer=False):
    def judge_empty_value(value):
        if value is None:
            return False
        if isinstance(value, str) and value.strip() == "":
            return False
        if isinstance(value, list) and len(value) == 0:
            return False
        return True
    inputs = []
    for key in _all_features:
        if judge_empty_value(_all_features[key]) and key not in ['id', 'description']:
            inputs.append("The feature [{}] is [{}].".format(key, _all_features[key]))
    if with_answer:
        answer = []
        for key in extract_data:
            if extract_data[key] is not None:
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
    # parser.add_argument('--model', type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1", help='model name')
    parser.add_argument('--model', type=str, default="mistralai/Mixtral-8x22B-Instruct-v0.1", help='model name')
    parser.add_argument("--max_tokens", type=int, default=64, help="max tokens for generation")
    args = parser.parse_args()
    all_features_but_desc, extracted_data = get_highlight_data()
    ids = list(extracted_data.keys())
    sample_train_data = 5
    random.seed(42)
    train_sample_ids_filename = "highlight_model_prompting_train_ids_{}shot.pt".format(sample_train_data)
    if os.path.exists(train_sample_ids_filename):
        train_ids = torch.load(train_sample_ids_filename)
    else:
        train_ids = random.sample(ids, sample_train_data)
        torch.save(train_ids, train_sample_ids_filename)
    train_sample_prompts = ["Your task is to identify the features highlighted in the following text given all the features. Here are some examples:\n\n"]
    for id_i, _id in enumerate(train_ids):
        _all_features = all_features_but_desc[_id]
        extracted_features = extracted_data[_id]
        prompt = format_sample_data(_all_features, extracted_features, with_answer=True)
        train_sample_prompts.append("---Sample {}---\n\n".format(id_i) + prompt)
    train_prompt = "".join(train_sample_prompts)
    print(train_prompt)

    prompts = []
    test_sample_ids = []
    ids_count = 0
    for _id in ids:
        if _id in train_ids:
            continue
        _all_features = all_features_but_desc[_id]
        extracted_features = extracted_data[_id]
        prompt = format_sample_data(_all_features, extracted_features)
        for key in DESIRED_FEATURE_NAMES:
            # if _all_features[key] is not None:
            prompts.append(train_prompt +
                           "Now, you are given a new home listing with the following features: \n\n" +
                           prompt +
                           "Can you tell me whether the feature [{}] should be highlighted in the text? Please answer by YES or NO.".format(key))
            test_sample_ids.append((_id, key))
        ids_count += 1
        if ids_count >= 100:
            break
    api_key = "[your-key]"
    os.environ['OPENAI_API_KEY'] = api_key
    # chat_manager = OpenAIChatManager(api_key)
    chat_manager = OpenAI(api_key=api_key, organization='org-nXJFZp9biBJ8I9Xtlbihfo0T',)
    # llm = LLM(model=args.model, tensor_parallel_size=4, gpu_memory_utilization=0.8, max_logprobs=10, seed=42)
    # sampling_params = SamplingParams(n=1, max_tokens=args.max_tokens, logprobs=10)
    # split ids into batches, after each batch, save the results
    num_batches = 10
    batch_size = len(prompts) // num_batches
    batch_ids = list(range(num_batches))
    # first search for existing files
    output_root_dir = "prompting_baseline_outputs/{}".format(os.path.basename(args.model))
    os.makedirs(output_root_dir, exist_ok=True)
    for i in range(num_batches):
        if os.path.exists(os.path.join(output_root_dir, "highlight_model_prompting_output_{}.pt".format(i))):
            batch_ids.remove(i)
    for i in batch_ids:
        start = i * batch_size
        end = (i + 1) * batch_size
        if i == num_batches - 1:
            end = len(prompts)
        # responses = llm.generate(prompts[start:end], sampling_params, use_tqdm=True)
        responses = []
        for prompt in tqdm(prompts[start:end]):
            response = chat_manager.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {'role': "system", "content": "You are a helpful assistant."},
                    {'role': "user", "content": prompt}
                ]
            )
            responses.append(response)
            # wait for 1 second
            time.sleep(1)
        torch.save([responses, test_sample_ids[start:end], prompts[start:end]], os.path.join(output_root_dir, "highlight_model_prompting_gpt4_output_{}.pt".format(i)))
    #
    #
    # responses = llm.generate(prompts, sampling_params, use_tqdm=True)
    # torch.save([responses, test_sample_ids, prompts], "highlight_model_prompting_output.pt")

