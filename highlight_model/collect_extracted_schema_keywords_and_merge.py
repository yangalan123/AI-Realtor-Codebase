import argparse
import nltk
import os.path

import torch
from datasets import load_dataset
from loguru import logger
from vllm import LLM, SamplingParams
from collections import Counter
from itertools import chain
from nltk.corpus import wordnet

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RAGAgentArgsParser.')
    parser.add_argument('--model', type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1", help='model name')
    args = parser.parse_args()

    vocab_ckpt = "collected_vocab.pt"
    confusion_dict_basic = {
        "sq ft": "sqft",
    }
    single_word_confusion = dict()
    if not os.path.exists(vocab_ckpt):
        lemmatizer = nltk.WordNetLemmatizer()
        # ckpt_name = "highlight_model_extracted_features_all_data_v5_per_word_two_stage_update_first_stage_prompt.pt"
        # v5 outputs looks not good, switch back to v4
        # ckpt_name = "highlight_model_extracted_features_all_data_v4_per_word_two_stage.pt"
        ckpt_name = "highlight_model_extracted_features_all_data_v6_per_word_two_stage_update_first_stage_prompt_default_case_instruction.pt"
        idx, outputs = torch.load(ckpt_name)
        collected_outputs = Counter()
        for output in outputs:
            if isinstance(output.outputs[0].text_normalized, list):
                processed_elements = [x.strip() for x in output.outputs[0].text_normalized]
            else:
                processed_elements = [x.strip() for x in output.outputs[0].text_normalized.split(", ")]
            final_buffer = []
            for i in range(len(processed_elements)):
                element = processed_elements[i].lower()
                if len(element) == 0:
                    continue
                if element[-1] == ".":
                    element = element[:-1]
                if "\n\n" in element:
                    element = element.split("\n\n")[0]
                if "\n\n" not in element and "\n" in element:
                    element = element.replace("\n", " ")
                if "output:" in element:
                    element = element.replace("output", "")
                if ", " in element:
                    element = element.split(", ")[0]
                if "-" in element:
                    element = element.split("-")[0]
                if " " not in element:
                    # single word, perform nltk lemmatization
                    element = lemmatizer.lemmatize(element)
                    standalone_flag = True
                    parent_element = None
                    if element not in single_word_confusion:
                        for key in single_word_confusion:
                            synonyms = wordnet.synsets(key)
                            lemmas = set(chain.from_iterable([word.lemma_names() for word in synonyms]))
                            if element in lemmas:
                                # single_word_confusion[element] = key
                                standalone_flag = False
                                parent_element = key
                                break
                        if standalone_flag:
                            single_word_confusion[element] = [element, ]
                        else:
                            single_word_confusion[parent_element].append(element)
                    if not standalone_flag:
                        element = parent_element

                if element in confusion_dict_basic:
                    element = confusion_dict_basic[element]
                if len(element) == 0:
                    continue
                final_buffer.append(element.strip())
                # final_buffer.append(element)


            collected_outputs.update(set(final_buffer))
        torch.save([collected_outputs, confusion_dict_basic, single_word_confusion], vocab_ckpt)
    else:
        collected_outputs, confusion_dict_basic, single_word_confusion = torch.load(vocab_ckpt)

    print("Extract {} Unique Keywords".format(len(collected_outputs)))
    print("Most Common Examples: ", collected_outputs.most_common(50))
    print("Least Common Examples: ", collected_outputs.most_common()[-50:])
    items = collected_outputs.most_common()
    word_count = 0
    token_counts_rough = 0
    word_list = []
    for item in items:
        k, v = item
        if v >= 50:
            word_count += 1
            token_counts_rough += len(k.split())
            word_list.append(k)
    print("word count", word_count)
    print("token_counts_rough", token_counts_rough)
    with open("collect_vocab.out.high_freq_50", "w") as f_out:
        f_out.write(", ".join(word_list))

        #if "transportation" in k or "proximity" in k:
            #if v >= 100:
                #print(k, v)

        # collected_outputs.append(output.outputs[0].text.strip())

    # llm = LLM(model=args.model, tensor_parallel_size=4, gpu_memory_utilization=0.8, seed=42, max_num_seqs=128)
    # sampling_params = SamplingParams(n=1, max_tokens=args.max_tokens, top_p=0.95)
