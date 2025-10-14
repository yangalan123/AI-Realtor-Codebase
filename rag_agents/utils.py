import json


def judge_empty_value(value):
    if value is None:
        return False
    if isinstance(value, str) and value.strip() == "":
        return False
    if isinstance(value, list) and len(value) == 0:
        return False
    return True

def get_zillow_data():
    filepath = "./data/housing_zillow.json"
    with open(filepath, "r", encoding='utf-8') as f_in:
        data = json.load(f_in)
        out_text_data = []
        for datum in data:
            features = datum['features']
            metadata = datum['metadata']
            template = (f"This is a house with {features['Bedrooms']} bedrooms and {features['Bathrooms']} bathrooms. "
                        f"It was built in {features['Year Built']} and is a {features['Home Type']} type of home. "
                        f"The average school rating in the area is {metadata['avg_school_rating']}. "
                        f"Here is a description of home features: {features['description']}. "
                        f"The price for this home is {metadata['price']}. The address is {metadata['street_address']}\n{metadata['regionString']}.")
            out_text_data.append(template)
    return out_text_data, data


def load_json_or_jsonl_data(filepath):
    try:
        with open(filepath, "r", encoding='utf-8') as f_in:
            data = json.load(f_in)
        return data
    except:
        data = []
        with open(filepath, "r", encoding='utf-8') as f_in:
            for line in f_in:
                data.append(json.loads(line))
        return data


def get_input_data(filepath="./data/input.json"):
    return load_json_or_jsonl_data(filepath)


def get_retrieval_data(filepath="./data/ai_realtor_listing_data.json"):
    return load_json_or_jsonl_data(filepath)


def factor_to_string(k: str, v0: int, v1: str) -> str:
    return k + " is " + {0: "more important", 1: "less important", 2: "indifferent"}[
        v0] + ", my criteria is " + v1 + '\t'


def get_annotator_preference_context():
    ## Chenghao
    # print(
    #     predict_preference_ranking(
    #         [0, 3],
    preference_context_dict = {}
    preference_context_dict["[name1]"] = {
        "original_dict": {
            "price": (0, "less than 1m"),
            "location": (0, "proximity to grocery store, prefer north side of Chicago"),
            "home feature/amenity": (0, "has garage, no annoying pipe, up-to-date facilities"),
            "house size": (1, "has at least 2b"),
            "investment value": (2, "good home condition, new roof"),
        },
    }
    preference_context_dict['[name1]']['factor_str'] = ''.join(
        [factor_to_string(k, v0, v1) for k, (v0, v1) in preference_context_dict['[name1]']['original_dict'].items()])

    preference_context_dict["[name2]"] = {
        "original_dict": {
            "price": (0, "less than 250k"),
            "location": (2, ""),
            "home feature/amenity": (1, "in-room laundry"),
            "house size": (0, ""),
            "investment value": (0, "good home condition, easy resale"),
        },
    }
    preference_context_dict['[name2]']['factor_str'] = ''.join(
        [factor_to_string(k, v0, v1) for k, (v0, v1) in preference_context_dict['[name2]']['original_dict'].items()])
    preference_context_dict["[name3]"] = {
        "original_dict": {
            "price": (0, "less than 1m"),
            "location": (0, "proximity to grocery store"),
            "home feature/amenity": (1, "has garage"),
            "house size": (1, "has at least 2b"),
            "investment value": (1, "good home condition, new roof"),
        },
    }
    preference_context_dict['[name3]']['factor_str'] = ''.join(
        [factor_to_string(k, v0, v1) for k, (v0, v1) in preference_context_dict['[name3]']['original_dict'].items()])

    preference_context_dict["[name4]"] = {
        "original_dict": {
            "price": (1, ""),
            "location": (0, ""),
            "home feature/amenity": (1, ""),
            "house size": (0, ""),
            "investment value": (2, ""),
        },
    }
    preference_context_dict['[name4]']['factor_str'] = ''.join(
        [factor_to_string(k, v0, v1) for k, (v0, v1) in preference_context_dict['[name4]']['original_dict'].items()])

    return preference_context_dict
