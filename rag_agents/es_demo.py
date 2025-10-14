import argparse
import copy
import json
import os
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams

from utils import get_input_data, get_retrieval_data

def create_index(es, index_name, settings):
    # Check if the index already exists
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, body=settings)
    else:
        print(f"Index '{index_name}' already exists.")

def index_documents(es, index_name, documents, schema_keys):
    # Validate documents against schema and track progress
    actions = []
    for doc in tqdm(documents, desc="Validating and preparing documents"):
        # Validate document fields against schema
        new_doc = copy.copy(doc)
        new_doc['area'] = doc['living_area']
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

def search(es, index_name, query_dict, num_results=3):
    # Convert user query to Elasticsearch query DSL
    must_clauses = [{"match": {field: query_dict[field]}} for field in query_dict if query_dict[field]]

    query_body = {
        "query": {
            "bool": {
                "must": must_clauses
            }
            # "match": {field: query_dict[field] for field in query_dict if query_dict[field]}
            # "match": {"bedroom": query_dict["bedroom"]}
            # "match_all": {}
        },
        "sort": [
            {"favorite_count": {"order": "desc"}}
        ],
        "size": num_results
    }
    response = es.search(index=index_name, body=query_body)
    return response['hits']['hits']

if __name__ == '__main__':
    es = Elasticsearch(["https://localhost:9200/",], basic_auth=("[username]", "[auth]"),verify_certs=False, ssl_show_warn=False)

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
                "page_view_count": {"type": "float"},
                "favorite_count": {"type": "float"},
                "home_insights": {"type": "keyword"},
                "neighborhood_region": {"type": "keyword"},
                "id": {"type": "keyword"}
            }
        }
    }

    schema_keys = set(index_settings['mappings']['properties'].keys())
    create_index(es, "real_estate", index_settings)

    documents = get_retrieval_data("./data/ai_realtor_listing_data.json")
    index_documents(es, "real_estate", documents, schema_keys)
    es.indices.refresh(index="real_estate")

    query = {
        "bedrooms": 3.0,
        "bathrooms": 2.0,
    }

    results = search(es, "real_estate", query)
    print(results)
