import argparse
import copy
import json
import os

import torch
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from tqdm import tqdm
from vllm import LLM, SamplingParams

from utils import get_input_data, get_retrieval_data, get_annotator_preference_context, judge_empty_value


def filter_data(doc, relevant_keys):
    return {key: value for key, value in doc.items() if key in relevant_keys and judge_empty_value(value)}

def augment_data(doc):
    # by default, we allow users to pass in a tuple to specify the min and maximum value
    key_list = list(doc.keys())
    for key in key_list:
        if key in ['bedrooms', 'bathrooms']:
            if isinstance(doc[key], float) or isinstance(doc[key], int):
                doc[key] = (doc[key], 100)
            else:
                doc[key] = tuple(doc[key])
        # elif key == "price":
        #     if isinstance(doc[key], float) or isinstance(doc[key], int):
        #         doc[key] = (0.0, doc[key])
        #     else:
        #         doc[key] = tuple(doc[key])
        # elif key == "area":
        #     if isinstance(doc[key], float) or isinstance(doc[key], int):
        #         doc[key] = (doc[key], 1e8)
        #     else:
        #         doc[key] = tuple(doc[key])

    return doc

def preprocess_data(doc):
    keys = set(doc.keys())
    for key in keys:
        if doc[key] is list:
            doc[key] = ",".join([str(x) for x in doc[key]])
    if "living_area" in doc and "area" not in doc:
        # from source doc, safely ignore it
        doc['area'] = doc['living_area']
    if "living_area_value" in doc and "area" not in doc:
        doc['area'] = doc['living_area_value']
    if "neighborhood" in doc and "neighborhood_region" not in doc:
        doc['neighborhood_region'] = doc['neighborhood']
        del doc['neighborhood']
    if "address" in doc and "street_address" not in doc:
        doc['street_address'] = doc['address']
        del doc['address']
    if "ID" in keys:
        del doc['ID']
    if "url" in keys:
        del doc['url']

    return doc


def create_index(es, index_name, settings):
    # Check if the index already exists
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, body=settings)
        return True
    else:
        print(f"Index '{index_name}' already exists.")
        return False


def index_documents(es, index_name, documents, schema_keys):
    # Validate documents against schema and track progress
    actions = []
    for doc in tqdm(documents, desc="Validating and preparing documents"):
        # Validate document fields against schema
        # new_doc = copy.copy(doc)
        # new_doc['area'] = doc['living_area']
        new_doc = preprocess_data(doc)
        if not schema_keys.issubset(new_doc.keys()):
            missing_keys = schema_keys - set(new_doc.keys())
            print(f"Warning: Document ID {new_doc.get('id', 'Unknown')} is missing keys: {missing_keys}")
            continue  # Skip adding this document to the bulk actions

        action = {
            "_index": index_name,
            "_id": new_doc["id"],
            "_source": new_doc,
        }
        actions.append(action)

    # Perform bulk indexing with progress bar
    tqdm(desc="Indexing documents", total=len(actions))
    bulk(es, actions)


def search(es, index_name, query_dict, num_results=50):
    # Convert user query to Elasticsearch query DSL
    # should_clauses = [{"match": {field: query_dict[field]}} for field in query_dict if judge_empty_value(query_dict[field])]
    augmented_query_dict = augment_data(query_dict)
    should_clauses = []
    filter_clauses = []
    for field, value in augmented_query_dict.items():
        if field in ['state', "city", "home_type"]:
            filter_clauses.append({"term": {field: value}})
            continue
        if isinstance(value, list):
            if len(value) > 0:
                should_clauses.append({"terms": {field: value}})
        elif isinstance(value, float) or isinstance(value, int):
            # should_clauses.append({"match": {field: value}})
            should_clauses.append({
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "source": "Math.abs(params.query_value - doc['{}'].value)".format(field),
                        "params": {
                            "query_value": value
                        }
                    },
                }
            })
        elif isinstance(value, tuple) and (isinstance(value[0], float) or isinstance(value[0], int)) and len(value) == 2:
            # range search
            filter_clauses.append({"range": {field: {"gte": value[0], "lte": value[1]}}})

        elif value is not None:
            should_clauses.append({"match": {field: value}})
    bool_query = {
        "should": should_clauses,
    }
    if len(filter_clauses) > 0:
        bool_query["filter"] = filter_clauses
    query_body = {
        "query": {
            "bool": bool_query
            # "match": {field: query_dict[field] for field in query_dict if query_dict[field]}
            # "match": {"bedroom": query_dict["bedroom"]}
            # "match_all": {}
        },
        "sort": [
            {"favorite_count": {"order": "desc"}}
        ],
        "size": num_results
    }
    # print(query_body)
    response = es.search(index=index_name, body=query_body)
    return [x["_source"] for x in response['hits']['hits']], query_body


def generate_prompts(top_results, test_query, preference_str=None, highlight_str=None):
    additional_instruction_preference = '\nThe buyer preferences would also be provided and you should consider them when generating the description to make it more appealing.'
    additional_instruction_highlight = '\nYou should also consider the highlighted features when generating the description to make it more appealing.'
    prompt_instructions = f"""Your task is to improve or generate a descriptive text for a real estate listing. 
    Consider the information from the search query, as well as details from similar listings. 
    Ensure the text is clear, concise, and highlights appealing features, as you see in the similar listings.
    If no relevant samples provided, use your best guess.{additional_instruction_preference if preference_str is not None else ''}{additional_instruction_highlight if highlight_str is not None else ''}

    Search Query:"""
    # prompt_instructions = f"""Your task is to improve or generate a descriptive text for a real estate listing.
    # Consider the information from the search query, as well as details from similar listings.
    # Ensure the text is clear, concise, and highlights appealing features, as you see in the similar listings.
    # If no relevant samples provided, use your best guess.
    #
    # Search Query:"""

    for key, value in test_query.items():
        if key in ["id", "description"]:
            continue
        prompt_instructions += f"\n{key}: {value}"

    if len(top_results) > 0:
        prompt_instructions += """\nListing Examples:\n"""

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

    if preference_str is not None:
        prompt_instructions += f"""
        \nBuyer Preferences: {preference_str}
        """
    if highlight_str is not None:
        prompt_instructions += f"""
        \nHighlighted Features: {highlight_str}
        """

    prompt_instructions += """
    \nRefined/Generated Description (Ends by "\n\n"):
    """

    return prompt_instructions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RAGAgentArgsParser.')
    # parser.add_argument('--model', type=str, default="CohereForAI/c4ai-command-r-plus", help='model name')
    parser.add_argument('--model', type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1", help='model name')
    parser.add_argument("--index_name", type=str, default="real_estate", help="index name")
    parser.add_argument("--retrieval_ckpt", type=str, default="retrieval.ckpt", help="retrieval checkpoint")
    parser.add_argument("--no_retrieval", action="store_true", help="skip retrieval")
    parser.add_argument("--gpt4_retrieval", action="store_true", help="use gpt4-generated-content for retrieval")
    parser.add_argument("--max_tokens", type=int, default=350, help="max tokens for generation")
    parser.add_argument('--input_data', type=str, default="../data/input.json", help='input data file')
    parser.add_argument('--retrieval_data', type=str, default="../data/zillow_cleaned_v5_allfeatures.json",
                        help='retrieval data file')
    parser.add_argument('--preference', type=str, help='whose preference?', default=None)
    parser.add_argument('--use_highlight', action="store_true", help='use highlight')
    parser.add_argument("--no_llm_running", action="store_true", help="skip llm running (pure-cpu work, useful if you only want to run search engine)")
    parser.add_argument("--output_root_dir", type=str, default="rag_output", help="output root directory")
    args = parser.parse_args()
    test_input_data = get_input_data(args.input_data)
    # retrieval_database = get_retrieval_data(args.retrieval_data)

    es_url = os.environ.get("ELASTICSEARCH_URL", "https://localhost:9200/")
    es_username = os.environ.get("ELASTICSEARCH_USERNAME")
    es_password = os.environ.get("ELASTICSEARCH_PASSWORD")
    basic_auth = (es_username, es_password) if es_username and es_password else None
    es = Elasticsearch([es_url], basic_auth=basic_auth, verify_certs=False, ssl_show_warn=False)

    index_settings = {
        "mappings": {
            "properties": {
                "bedrooms": {"type": "float"},
                "bathrooms": {"type": "float"},
                "price": {"type": "float"},
                "description": {"type": "text"},
                "area": {"type": "float"},
                "street_address": {"type": "text"},
                "home_type": {"type": "keyword"},
                "state": {"type": "keyword"},
                "city": {"type": "keyword"},
                "page_view_count": {"type": "float"},
                "favorite_count": {"type": "float"},
                "home_insights": {"type": "keyword"},
                "neighborhood_region": {"type": "keyword"},
                "id": {"type": "keyword"}
            }
        }
    }

    schema_keys = set(index_settings['mappings']['properties'].keys())
    index_name = args.index_name
    new_index_flag = create_index(es, index_name, index_settings)

    documents = get_retrieval_data(args.retrieval_data)
    preference_context = get_annotator_preference_context()
    # output_root_dir = "rag_output"
    output_root_dir = args.output_root_dir
    os.makedirs(output_root_dir, exist_ok=True)

    if not args.no_retrieval:
        schema_keys = set(index_settings['mappings']['properties'].keys())
        # create_index(es, index_name, index_settings)
        # if new_index_flag:
        index_documents(es, index_name, documents, schema_keys)
        es.indices.refresh(index=index_name)
        # Create the index
        # index = create_in(args.index_name, schema)
        # with index.writer(procs=4, limitmb=256) as writer:
        #     for doc in tqdm(retrieval_database, desc="Indexing documents"):
        #         filtered_doc = filter_data(doc, relevant_keys)
        #         filtered_doc = postprocess_data(filtered_doc)
        #         writer.update_document(**filtered_doc)

    # llm = LLM(model=args.model, tensor_parallel_size=4)
    # sampling_params = SamplingParams(n=1, max_tokens=300)
    # response = llm.generate([prompt_instructions,], sampling_params)[0].outputs[0].text
    prompts = []
    return_results_num = []
    if not args.no_retrieval:
        zero_return_count = 0
        # ckpt_dir = "rag_ckpt"
        ckpt_dir = os.path.join(output_root_dir, "rag_ckpt")
        os.makedirs(ckpt_dir, exist_ok=True)
        # top_k = 3
        top_k = 10
        ckpt_name = "retrieval_top_{}_{}_es{}{}{}{}.ckpt".format(os.path.basename(args.input_data), top_k, "_{}_preference".format(
            args.preference) if args.preference is not None else "",
                                                          f"{'_no_retrieval' if args.no_retrieval else ''}",
                                                          f"{'_gpt4_gen_retrieve' if args.gpt4_retrieval else ''}",
                                                          f"{'_use_highlight' if args.use_highlight else ''}")
        ckpt_name = os.path.join(ckpt_dir, ckpt_name)
        retrieval_keys = schema_keys - {'id', 'description'}
        # retrieval_keys = {'bedrooms', "bathrooms", "home_type"}
        retrieved_results = []
        if os.path.exists(ckpt_name):
            prompts = torch.load(ckpt_name)
        else:
            for doc in tqdm(test_input_data, desc="Processing documents"):
                test_query = filter_data(preprocess_data(doc), retrieval_keys)
                results, es_query = search(es, index_name, test_query)
                if "id" in doc:
                    results = [x for x in results if x["id"] != doc["id"]]
                retrieved_results.append(results)
                if len(results) >= top_k:
                    results = results[:top_k]
                return_results_num.append(len(results))
                if len(results) == 0:
                    print("No results found for query: ", test_query)
                    print("ES query:", es_query)
                    zero_return_count += 1
                    #exit()
                if args.preference is not None:
                    preference_str = preference_context.get(args.preference, None)
                else:
                    preference_str = None
                if args.use_highlight:
                    assert "highlight_feature_model" in doc, "No highlight feature model found in the input data"
                    highlight_features = doc["highlight_feature_model"]
                    highlight_str = ", ".join(
                        [key for key, value in highlight_features.items() if
                         key not in ["id", "description"] and value >= 0.5])
                    highlight_str = highlight_str + "."
                else:
                    highlight_str = None
                prompt = generate_prompts(results, test_query, preference_str, highlight_str)
                # new_doc = copy.deepcopy(doc)
                # new_doc['description'] = new_description
                prompts.append(prompt)
            torch.save(prompts, ckpt_name)
        print(f"Zero return count: {zero_return_count} / {len(test_input_data)}")
        print("Average number of retrieved results: ", sum(return_results_num) / len(return_results_num))
        if zero_return_count == test_input_data:
            print("All queries return zero results, please check the input data and the retrieval database")
            # remove ckpt
            os.remove(ckpt_name)
            exit()
        if len(retrieved_results) > 0:
            print("Average number of retrieved results: ",
                  sum([len(x) for x in retrieved_results]) / len(retrieved_results))
            new_doc_buffers = []
            for doc_i in range(len(test_input_data)):
                new_doc = copy.deepcopy(test_input_data[doc_i])
                if not args.no_llm_running:
                    new_doc['retrieved_results'] = retrieved_results[doc_i]
                else:
                    new_doc['retrieved_results'] = [x['id'] for x in retrieved_results[doc_i]]
                new_doc_buffers.append(new_doc)
            ckpt_json_version_name = ckpt_name.replace(".ckpt", ".json")
            with open(ckpt_json_version_name, "w") as f_out:
                for new_doc in new_doc_buffers:
                    f_out.write(json.dumps(new_doc) + "\n")
    else:
        if args.gpt4_retrieval:
            retrieval_results = json.load(open("../data/output_bestshot.json", "r"))
        else:
            # no-retrieval baseline
            retrieval_results = []
        for doc in tqdm(test_input_data, desc="Processing documents"):
            if args.use_highlight:
                assert "highlight_feature_model" in doc, "No highlight feature model found in the input data"
                highlight_features = doc["highlight_feature_model"]
                highlight_str = ", ".join(
                    [key for key, value in highlight_features.items() if key != "id" and value >= 0.5])
                highlight_str = highlight_str + "."
            else:
                highlight_str = None
            test_query = filter_data(preprocess_data(doc), schema_keys)
            if args.preference is not None:
                preference_str = preference_context.get(args.preference, None)
            else:
                preference_str = None
            prompt = generate_prompts(retrieval_results, test_query, preference_str, highlight_str)
            # if len(prompt) == 0:
            #     # show the example prompt
            #     print("Example: \n", prompt)
            prompts.append(prompt)
            # by default, no need to save ckpt as it will be a relatively fast process

    if not args.no_llm_running:
        llm = LLM(model=args.model, tensor_parallel_size=4, gpu_memory_utilization=0.8)
        sampling_params = SamplingParams(n=1, max_tokens=args.max_tokens)
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
        with open(
                os.path.join(output_root_dir,
                             f"{os.path.basename(args.input_data).strip('.json')}"
                             f"_max_token_{args.max_tokens}"
                             f"_{os.path.basename(args.model)}"
                             f"{'_no_retrieval' if args.no_retrieval else ''}"
                             f"{'_gpt4_gen_retrieve' if args.gpt4_retrieval else ''}"
                             f"{'_use_highlight' if args.use_highlight else ''}"
                             f"{'_' + args.preference + '_preference' if args.preference is not None else ''}.json",
                             ), "w"
        ) as f_out:
            for doc_i, doc in tqdm(enumerate(test_input_data), desc="Processing documents"):
                new_doc = copy.deepcopy(doc)
                # new_doc['description'] = outputs[doc_i].outputs[0].text
                text = outputs[doc_i].outputs[0].text.strip()
                if "END" in text:
                    text = text[:text.index("END")].strip()
                if "\n\n" in text:
                    sents = text.split("\n\n")[0].split("\n")
                    sents = [d.strip() for d in sents if len(d.strip()) > 0]
                    sents = [d for d in sents if not all([c in ".,:;?!-" for c in d])]
                    text = " ".join(sents)
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
