import argparse
import copy
import json
import os

import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from whoosh import fields
from whoosh.fields import Schema
from whoosh.index import create_in
from whoosh.qparser import MultifieldParser

from utils import get_input_data, get_retrieval_data


def filter_data(doc, relevant_keys):
    return {key: value for key, value in doc.items() if key in relevant_keys}


def postprocess_data(doc):
    for key in doc:
        if doc[key] is list:
            doc[key] = ",".join([str(x) for x in doc[key]])
    return doc


def custom_scoring(searcher, docnum, score):
    return score * searcher.stored_fields(docnum)['favorite_count']


def search(query_dict, index, schema, num_results=3):
    # relevant_query = {key: value for key, value in query_dict.items() if key in schema.names()}
    relevant_query = filter_data(query_dict, schema.names())

    with index.searcher() as searcher:
        query_parser = MultifieldParser(list(relevant_query.keys()), schema=schema)
        # Construct the query
        search_terms = " ".join([str(x) for x in relevant_query.values()])
        # Add a boost based on favorite count (adjust '2.0' if needed)
        # if "favorite_count" in relevant_query:
        #     search_terms += f" ^2.0"
        query = query_parser.parse(search_terms)

        results = searcher.search(query, limit=num_results, sortedby="favorite_count")
        return results, relevant_query


def generate_prompts(top_results, test_query):
    prompt_instructions = """Your task is to improve or generate a descriptive text for a real estate listing. 
    Consider the information from the search query, as well as details from similar listings. 
    Ensure the text is clear, concise, and highlights appealing features, as you see in the similar listings.
    If no relevant samples provided, use your best guess based on the search query.

    Search Query:"""

    for key, value in test_query.items():
        prompt_instructions += f"\n    {key}: {value}"

    if len(top_results) > 0:
        prompt_instructions += """\nListing Examples:"""

        for result in top_results:
            prompt_instructions += """
            ---
            Description: {}
            Bedrooms: {}
            Bathrooms: {}
            Price: {}
            Neighborhood: {}
            ---
            """.format(
                result.get('description', 'No description available'),
                result.get('bedroom', ''),
                result.get('bathrooms', ''),
                result.get('price', ''),
                result.get('neighborhood_region', '')
            )

    prompt_instructions += """
    Refined/Generated Description (Ends by "END"):
    """

    return prompt_instructions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RAGAgentArgsParser.')
    # parser.add_argument('--model', type=str, default="CohereForAI/c4ai-command-r-plus", help='model name')
    parser.add_argument('--model', type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1", help='model name')
    parser.add_argument("--index_name", type=str, default="myindex", help="index name")
    parser.add_argument("--retrieval_ckpt", type=str, default="retrieval.ckpt", help="retrieval checkpoint")
    parser.add_argument("--no_retrieval", action="store_true", help="skip retrieval")
    parser.add_argument("--gpt4_retrieval", action="store_true", help="use gpt4-generated-content for retrieval")
    parser.add_argument("--max_tokens", type=int, default=350, help="max tokens for generation")
    parser.add_argument('--input_data', type=str, default="../data/input.json", help='input data file')
    parser.add_argument('--retrieval_data', type=str, default="../data/zillow_cleaned_v5_allfeatures.json", help='retrieval data file')
    args = parser.parse_args()
    test_input_data = get_input_data(args.input_data)
    retrieval_database = get_retrieval_data(args.retrieval_data)
    os.makedirs(args.index_name, exist_ok=True)

    # Define your schema
    schema = Schema(
        bedroom=fields.NUMERIC(stored=True, numtype=float),  # For precise sorting and filtering
        bathrooms=fields.NUMERIC(stored=True, numtype=float),
        price=fields.NUMERIC(stored=True, numtype=float),
        description=fields.TEXT(stored=True, vector=True),  # Enable full-text search and store term vectors
        area=fields.NUMERIC(stored=True, numtype=float),
        # area_units=fields.STORED(),  # For exact matching on units
        # brokerage_name=fields.KEYWORD(stored=True),
        # zipcode=fields.KEYWORD(stored=True),
        street_address=fields.TEXT(stored=True),  # If you need to search within the address
        # home_type=fields.KEYWORD(stored=True),
        page_view_count=fields.NUMERIC(stored=True, numtype=float),
        favorite_count=fields.NUMERIC(stored=True, numtype=float),  # Make favorite_count scorable
        home_insights=fields.KEYWORD(stored=True, commas=True),  # Index as comma-separated terms
        # neighborhood_region=fields.KEYWORD(stored=True),
        # time_on_zillow_days=fields.NUMERIC(stored=True, numtype=float),
        id=fields.ID(unique=True, stored=True),  # Unique identifier
        # score=fields.NUMERIC(stored=True, numtype=float)  # For storing a relevance/ranking score
    )

    relevant_keys = set(schema.names())

    if not args.no_retrieval:
        # Create the index
        index = create_in(args.index_name, schema)
        with index.writer(procs=4, limitmb=256) as writer:
            for doc in tqdm(retrieval_database, desc="Indexing documents"):
                filtered_doc = filter_data(doc, relevant_keys)
                filtered_doc = postprocess_data(filtered_doc)
                writer.update_document(**filtered_doc)

    # llm = LLM(model=args.model, tensor_parallel_size=4)
    # sampling_params = SamplingParams(n=1, max_tokens=300)
    # response = llm.generate([prompt_instructions,], sampling_params)[0].outputs[0].text
    prompts = []
    if not args.no_retrieval:
        zero_return_count = 0
        ckpt_name = "retrieval_{}.ckpt".format(os.path.basename(args.input_data))
        if os.path.exists(ckpt_name):
            prompts = torch.load(ckpt_name)
        else:
            for doc in tqdm(test_input_data, desc="Processing documents"):
                results, test_query = search(doc, index, schema)
                if len(results) == 0:
                    print("No results found for query: ", test_query)
                    zero_return_count += 1
                prompt = generate_prompts(results, test_query)
                # new_doc = copy.deepcopy(doc)
                # new_doc['description'] = new_description
                prompts.append(prompt)
            torch.save(prompts, ckpt_name)
        print(f"Zero return count: {zero_return_count}")
    else:
        if args.gpt4_retrieval:
            retrieval_results = json.load(open("../data/output_bestshot.json", "r"))
        else:
            # no-retrieval baseline
            retrieval_results = []
        for doc in tqdm(test_input_data, desc="Processing documents"):
            test_query = filter_data(doc, relevant_keys)
            prompt = generate_prompts(retrieval_results, test_query)
            # if len(prompt) == 0:
            #     # show the example prompt
            #     print("Example: \n", prompt)
            prompts.append(prompt)
            # by default, no need to save ckpt as it will be a relatively fast process

    llm = LLM(model=args.model, tensor_parallel_size=4, gpu_memory_utilization=0.8)
    sampling_params = SamplingParams(n=1, max_tokens=args.max_tokens)
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    with open(
            f"rag_output_{os.path.basename(args.input_data).strip('.json')}_max_token_{args.max_tokens}_{os.path.basename(args.model)}{'_no_retrieval' if args.no_retrieval else ''}{'_gpt4_gen_retrieve' if args.gpt4_retrieval else ''}.json",
            "w") as f_out:
        for doc_i, doc in tqdm(enumerate(test_input_data), desc="Processing documents"):
            new_doc = copy.deepcopy(doc)
            # new_doc['description'] = outputs[doc_i].outputs[0].text
            text = outputs[doc_i].outputs[0].text.strip()
            if "END" in text:
                text = text[:text.index("END")].strip()
            new_doc['description'] = text
            new_doc['prompt'] = outputs[doc_i].prompt
            f_out.write(json.dumps(new_doc) + "\n")

# Search
# with index.searcher() as searcher:
#     query_parser = QueryParser("content", schema)
#     query = query_parser.parse("language")
#     results = searcher.search(query)
#
#     for hit in results:
#         print(hit['id'], hit['title'])
