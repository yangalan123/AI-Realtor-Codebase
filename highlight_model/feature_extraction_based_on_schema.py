import argparse
import copy
import os.path
import random

import torch
from datasets import load_dataset
from loguru import logger
from vllm import LLM, SamplingParams
from utils import load_schema, load_hf_dataset
import json


if __name__ == '__main__':
    logger.add("feature_extraction_v10.log", rotation="10 MB")
    parser = argparse.ArgumentParser(description='RAGAgentArgsParser.')
    parser.add_argument('--model', type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1", help='model name')
    # parser.add_argument('--model', type=str, default="mistralai/Mixtral-8x22B-Instruct-v0.1", help='model name')
    parser.add_argument("--max_tokens", type=int, default=32, help="max tokens for generation")
    parser.add_argument("--checkpoint_freq", type=int, default=20000, help="checkpoint frequency")
    parser.add_argument("--schema", type=str, default="claude_output/claude_merged.json", help="schema file")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--total_splits", type=int, default=5, help="total splits")
    parser.add_argument("--split_id", type=int, default=0, help="split id")
    parser.add_argument("--rating_by_score", action="store_true", help="rating by score")
    parser.add_argument("--top_k", type=int, default=2000, help="top k")
    parser.add_argument("--filter_by_city", type=str, default="", help="filter by city")
    args = parser.parse_args()
    logger.info(args)
    random.seed(args.seed)
    model_name = os.path.basename(args.model)
    # ckpt_rootname = f"highlight_model_{model_name}_extracted_features_all_data_v8_claude_schema_after_human_annotation"
    ckpt_rootname = "highlight_model_{}_extracted_features_all_data_v10_claude_schema_after_human_annotation{}{}".format(
        model_name,
        f"_filterby_{args.filter_by_city}" if len(args.filter_by_city) > 0 else "",
        f"_top_{args.top_k}" if args.rating_by_score else "")
    os.makedirs(ckpt_rootname, exist_ok=True)
    ckpt_prompt_name = f"{ckpt_rootname}/prompts.pt"
    if not os.path.exists(ckpt_prompt_name):
        dataset = load_hf_dataset("[hf-path]", args.rating_by_score, args.top_k, args.filter_by_city)
        # df = dataset['train'].to_pandas()
        df = dataset.to_pandas()
        descriptions = df['description'].tolist()
        scores = df['score'].tolist()
        score_mapping = []
        total_prompts = []
        schema_dict = load_schema(args.schema)
        logger.info("#(categories):{}".format(len(list(schema_dict.keys()))) )
        for desc_i, desc in enumerate(descriptions):
            start_id = len(total_prompts)
            for category in schema_dict:
                sampled_keywords = random.sample(schema_dict[category], min(10, len(schema_dict[category])))
                total_prompts.append(
                    "Your task is to determine whether the given label class category is mentioned in the description. The meaning of the category will be explained by example keywords in this category. Only respond by 'YES' or 'NO'. "
                    "Label Class Category: {}. \n\nExample Keywords in this Category: {}\n\n"
                    "\n\nDescription: {}\n\nResponse (Yes/No): ".format(category, sampled_keywords, desc))
            score_mapping.append((scores[desc_i], start_id, len(total_prompts)))

        torch.save([total_prompts, scores, score_mapping], ckpt_prompt_name)
        logger.info("Saved {} total prompts to {}".format(len(total_prompts), ckpt_prompt_name))
    else:
        try:
            total_prompts, scores, score_mapping = torch.load(ckpt_prompt_name)
        except:
            # old version compatibility
            total_prompts = torch.load(ckpt_prompt_name)
            scores = None
        logger.info("Loaded {} prompts from {}".format(len(total_prompts), ckpt_prompt_name))
    # divide the prompts into splits
    num_prompts = len(total_prompts)
    prompts_per_split = num_prompts // args.total_splits
    prompts = total_prompts[args.split_id * prompts_per_split: (args.split_id + 1) * prompts_per_split]
    logger.info("Split id: {}, total prompts: {}, prompts per split: {}".format(args.split_id, len(prompts), prompts_per_split))

    ckpt_name = "{}/ckpt.pt.{}".format(ckpt_rootname, args.split_id)
    outputs = []
    if os.path.exists(ckpt_name):
        _prompts, outputs = torch.load(ckpt_name)
        assert _prompts == prompts, "Prompts mismatch"
    # llm = LLM(model=args.model, tensor_parallel_size=4, gpu_memory_utilization=0.8, seed=42, max_num_seqs=32)
    # llm = LLM(model=args.model, tensor_parallel_size=4, gpu_memory_utilization=0.9, seed=42)
    llm = LLM(model=args.model, tensor_parallel_size=2, gpu_memory_utilization=0.95, seed=42, max_model_len=16384)
    sampling_params = SamplingParams(n=1, max_tokens=args.max_tokens, top_p=0.9)
    # split prompts into batches and save the results after each batch
    # batch_size == checkpoint_freq
    logger.info("Start from index: {}".format(len(outputs)))
    for i in range(len(outputs), len(prompts), args.checkpoint_freq):
        batch_prompts = prompts[i: i + args.checkpoint_freq]
        batch_outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=True)
        # output_text = [output.outputs[0].text.strip() for output in batch_outputs]
        zipped_outputs = []
        for output_inst_i, output_inst in enumerate(batch_outputs):
            new_output_item = {
                "finished": output_inst.finished,
                "outputs": [x.text for x in output_inst.outputs],
                "cumulative_logprob": [x.cumulative_logprob for x in output_inst.outputs],
                "stop_reason": [x.stop_reason for x in output_inst.outputs],
                "finish_reason": [x.finish_reason for x in output_inst.outputs],
                "prompt": output_inst.prompt,
            }
            zipped_outputs.append(new_output_item)
            # drop all other irrelevant statistics
        outputs.extend(zipped_outputs)
        torch.save((prompts, outputs), ckpt_name)
        logger.info("Saved checkpoint at index: {}, Progress: {}".format(i, i / len(prompts) * 100))
    logger.info("Finished all prompts.")
