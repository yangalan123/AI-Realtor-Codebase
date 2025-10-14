import argparse
import os.path

import torch
from datasets import load_dataset
from loguru import logger
from vllm import LLM, SamplingParams

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RAGAgentArgsParser.')
    parser.add_argument('--model', type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1", help='model name')
    # parser.add_argument('--model', type=str, default="mistralai/Mixtral-8x22B-Instruct-v0.1", help='model name')
    parser.add_argument("--max_tokens", type=int, default=256, help="max tokens for generation")
    parser.add_argument("--checkpoint_freq", type=int, default=1000, help="checkpoint frequency")
    args = parser.parse_args()
    dataset = load_dataset("Sigma-Lab/zillow_cleaned_v5_allfeatures_rev2")
    df = dataset['train'].to_pandas()
    descriptions = df['description']
    prompts = []
    for desc in descriptions:
        # prompts.append(
        #     "Your task is to extract attractive highlight **Category** and do not add quantifiers, numbers, adjective or any modifiers. "
        #     "(e.g., 'two bedrooms/4-bedroom/Three Bedrooms' IS NOT Acceptable, but 'bedrooms' is okay. 'roof' is okay but 'new roof/updated roof' is NOT Acceptable). "
        #     "Please express categories as phrases or keywords) from the following house description. Each category should be separated by a comma. "
        #     "\n\nDescription: {}\n\nHighlights: ".format(desc))
        prompts.append(
            "Your task is to extract attractive highlights. "
            "(e.g., 'modern amenities', 'great views', 'lush landscaping', 'bamboo flooring'). "
            "Please express these highlights as phrases or keywords from the following house description. Each highlight should be separated by a comma. "
            "\n\nDescription: {}\n\nHighlights: ".format(desc))
    # ckpt_name = "highlight_model_extracted_features_all_data_v2_category.pt"
    # ckpt_name = "highlight_model_extracted_features_all_data_v3_two_stage.pt"
    # ckpt_name = "highlight_model_extracted_features_all_data_v4_per_word_two_stage.pt"
    # ckpt_name = "highlight_model_extracted_features_all_data_v5_per_word_two_stage_update_first_stage_prompt.pt"
    ckpt_name = "highlight_model_extracted_features_all_data_v6_per_word_two_stage_update_first_stage_prompt_default_case_instruction.pt"
    outputs = []
    idx = 0
    if os.path.exists(ckpt_name):
        idx, outputs = torch.load(ckpt_name)
    # llm = LLM(model=args.model, tensor_parallel_size=4, gpu_memory_utilization=0.8, seed=42, max_num_seqs=32)
    llm = LLM(model=args.model, tensor_parallel_size=4, gpu_memory_utilization=0.8, seed=42, max_num_seqs=128)
    sampling_params = SamplingParams(n=1, max_tokens=args.max_tokens, top_p=0.9)
    # split prompts into batches and save the results after each batch
    # batch_size == checkpoint_freq
    logger.info("Start from index: {}, len(outputs)={}".format(idx, len(outputs)))
    for i in range(idx, len(prompts), args.checkpoint_freq):
        batch_prompts = prompts[i: i + args.checkpoint_freq]
        batch_outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=True)
        output_text = [output.outputs[0].text.strip() for output in batch_outputs]
        # new_prompts = ["Please remove the quantifiers, numbers, adjectives or any modifiers from the following highlights, each highlight is separated by a comma. \n\nHighlights: {}\n\nNormalized Highlights: ".format(text) for text in output_text]
        new_prompts = []
        for text_i, _text in enumerate(output_text):
            phrases = [x.strip() for x in _text.split(",")]
            new_prompt = ("Please remove the quantifiers, numbers, adjectives or any modifiers in the provided phrase. "
                          "Uppercase or lowercase doesn't matter. "
                          "If the given input is already precise enough, please provide the same input."
                          "If you are not sure what to do, please also provide the input as it is. "
                          "Do not explain or provide additional information."
                          "Here are a few examples:"
                          "\n\nInput: Two Bedrooms.\n\nOutput: Bedrooms."
                          "\n\nInput: Newly Renovated Kitchen.\n\nOutput: Kitchen."
                          "\n\nInput: landscape. \n\nOutput: landscape."
                          "[Example Ends]"
                          "Now, given the Input, please precisely provide the Output."
                          "\n\nInput: {}\n\nOutput (should only be a noun phrase or keyword): ")
            _new_prompts = [(new_prompt.format(phrase), text_i) for phrase in phrases]
            new_prompts.extend(_new_prompts)
        # new_prompts = [
        #     "Please remove the quantifiers, numbers, adjectives or any modifiers in the given phrase. \n\nHighlights: {}\n\nNormalized Highlights: ".format(
        #         text) for text in output_text]
        batch_outputs_normalized = llm.generate([x[0] for x in new_prompts], sampling_params, use_tqdm=True)
        # redistribute the normalized text to the original prompts
        normalized_phrase_dict = dict()
        normalized_prompts_dict = dict()
        for j in range(len(batch_outputs_normalized)):
            original_text_i = new_prompts[j][1]
            if original_text_i not in normalized_phrase_dict:
                normalized_phrase_dict[original_text_i] = []
                normalized_prompts_dict[original_text_i] = []
            normalized_phrase_dict[original_text_i].append(batch_outputs_normalized[j].outputs[0].text)
            normalized_prompts_dict[original_text_i].append(new_prompts[j][0])
        for j in range(len(batch_outputs)):
            # batch_outputs[j].outputs[0].text_normalized = ", ".join(normalized_phrase_dict[j])
            batch_outputs[j].outputs[0].text_normalized = normalized_phrase_dict[j]
            batch_outputs[j].prompt_normalized = normalized_prompts_dict[j]
            # batch_outputs[j].outputs[0].text_normalized = batch_outputs_normalized[j].outputs[0].text
            # batch_outputs[j].prompt_normalized = new_prompts[j]
        # for j in range(len(batch_outputs_normalized)):
        #     batch_outputs[j].outputs[0].text_normalized = batch_outputs_normalized[j].outputs[0].text
        #     batch_outputs[j].prompt_normalized = new_prompts[j]
        outputs.extend(batch_outputs)
        torch.save((i, outputs), ckpt_name)
        logger.info("Saved checkpoint at index: {}, Progress: {}".format(i, i / len(prompts) * 100))
