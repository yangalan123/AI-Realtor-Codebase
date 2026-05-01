import os
import json
from collections import Counter
import traceback
import pickle
import csv
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm
import numpy as np

class MainInfo(BaseModel):
    price_mentioned: bool
    price: float
    # home_insights_mentioned: bool
    # home_insights: list[str]
    living_area_mentioned: bool
    living_area: str
    bedrooms_mentioned: bool
    bedrooms: float
    bathrooms_mentioned: bool
    bathrooms: float
    # address_mentioned: bool
    # address: str

import re

def calculate_accuracy(extracted_info, main_info):
    correct = 0
    count = 0
    mismatched_keys = []
    for key, value in extracted_info.items():
        if "mentioned" in key:
            continue
        if not extracted_info[f"{key}_mentioned"]:
            continue
        if key in ["home_insights", "address"] or key not in main_info:
            continue
        if value is not None:
            if isinstance(value, str):
                if value.lower() not in ["null", "not specified"] and len(value) > 0:
                    if key == "living_area":
                        if "0.0" in main_info[key] or "nan" in main_info[key]:
                            # original information broken, skip
                            continue
                        # Check if the value matches the pattern: float_number + [optional spaces] + 'sq' + ...
                        pattern = r'^\d+(\.\d+)?\s*sq'
                        if re.match(pattern, value.lower()):
                            _value = value.split("sq")[0].strip()
                            if _value.lower() in main_info[key].lower() or main_info[key].lower() in _value.lower():
                                correct += 1
                            else:
                                mismatched_keys.append({"key": key, "extracted_value": value, "original_value": main_info[key]})
                            count += 1
                        else:
                            # Skip if the pattern doesn't match
                            continue
                    else:
                        if value.lower() in main_info[key].lower() or main_info[key].lower() in value.lower():
                            correct += 1
                        else:
                            mismatched_keys.append({"key": key, "extracted_value": value, "original_value": main_info[key]})
                        count += 1
            else:
                if key in ["bedrooms", "bathrooms"]:
                    if abs(value - main_info[key]) <= 0.5:
                        correct += 1
                    else:
                        mismatched_keys.append({"key": key, "extracted_value": value, "original_value": main_info[key]})
                    count += 1
                else:
                    if abs(value) < 1e-6 and key == "price":
                        # in most cases, this means the model cannot extract the value
                        continue
                    if abs(value - main_info[key]) <= 0.1 * main_info[key]:
                        correct += 1
                    else:
                        mismatched_keys.append({"key": key, "extracted_value": value, "original_value": main_info[key]})
                    count += 1
    return correct, count, mismatched_keys

def write_mismatch_csv(model, mismatches):
    escaped_model = model.replace(" ", "_")
    escaped_model = escaped_model.replace("+", "_")
    filename = f"{escaped_model}_mismatches.csv"
    fieldnames = ['description', 'key', 'extracted_value', 'original_value']
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for mismatch in mismatches:
            writer.writerow(mismatch)

if __name__ == '__main__':
    client = OpenAI()
    if os.path.exists("model_wise_responses.pkl"):
        model_wise_responses = pickle.load(open("model_wise_responses.pkl", "rb"))
    else:
        model_wise_responses = dict()
        with open("responses_latest.json", "r", encoding='utf-8') as f:
            data = json.load(f)
            for user in data:
                responses = user["responses"]
                for response_i, response in enumerate(responses):
                    models = response["models"]
                    descriptions = response["descriptions"]
                    selection = response["selection"]
                    selected_model = models[selection]
                    rating = response["rating"]
                    main_info = response['listing']
                    main_info.pop("listing_id")
                    main_info['price'] = float(main_info['price'])
                    for _model, description in zip(models, descriptions):
                        if _model not in model_wise_responses:
                            model_wise_responses[_model] = []
                        model_wise_responses[_model].append({
                            "description": description,
                            "user_id": user["id"],
                            "response_i": response_i,
                            "selected": _model == selected_model,
                            "main_info": main_info,
                            "rating": rating
                        })
        pickle.dump(model_wise_responses, open("model_wise_responses.pkl", "wb"))
    ret = dict()
    metrics = dict()
    checked = dict()
    mismatch = dict()
    ckpt_filename = "extracted_info_is_mentioned_paper_version.pkl"
    if os.path.exists(ckpt_filename):
        ret, metrics, checked = pickle.load(open(ckpt_filename, "rb"))

    model_pbar = tqdm(model_wise_responses.items(), desc="Processing models", total=len(model_wise_responses),
                      position=0)
    for model, responses in model_pbar:
        model_pbar.set_description(f"Processing model {model}")
        mismatched_counters = Counter()
        model_mismatches = []
        if model not in checked:
            ret[model] = []
            metrics[model] = [[], []]
            checked[model] = set()

        response_pbar = tqdm(enumerate(responses), total=len(responses), desc=f"Processing responses", position=1,
                             leave=False)
        for response_i, response in response_pbar:
            try:
                if response_i in checked[model]:
                    extracted_info = ret[model][response_i]["extracted_info"]
                    assert ret[model][response_i]["user_id"] == response["user_id"]
                    assert ret[model][response_i]["response_i"] == response["response_i"]
                    assert ret[model][response_i]["selected"] == response["selected"]
                    assert ret[model][response_i]["main_info"] == response["main_info"]
                    assert ret[model][response_i]["rating"] == response["rating"]
                    correct, count, mismatched_keys = calculate_accuracy(extracted_info, response["main_info"])
                else:
                    # example_home_insights =["Large island", "Oversized bathroom", "Open floor plan", "Lake views", "Orange l lines", "Newer stainless steel appliances", "Gorgeous hardwood floors", "Tons of cabinet space", "In-unit washer and dryer", "Skyline view", "Private balcony", "Beautiful city"]
                    # example_addr = "1255 S State St UNIT 703 Chicago IL 60601"
                    completion = client.beta.chat.completions.parse(
                        model="gpt-4o-mini-2024-07-18",
                        messages=[
                            # {"role": "system", "content": "Extract Real Estate Information. "
                            #                               "Find the price (e.g, 290000.0), home insights (e.g., {}), living area (e.g., '990.0 sqft'), bedrooms (e.g., 2), bathrooms (e.g., 3), and address (e.g., {}) from the description. "
                            #                               "Not all information may be present, so you also have to determine whether each field is mentioned or not.".format(example_home_insights, example_addr)},
                            {"role": "system", "content": "Extract Real Estate Information. "
                                                          "Find the price (e.g, 290000.0, take care we need to extract the price for whole real estate, and sometimes the description may only mention the price of some parts. In that case, mark this item as not mentioned), living area (e.g., '990.0 sqft'), bedrooms (e.g., 2), and bathrooms (e.g., 3) from the description. "
                                                          "Not all information may be present, so you also have to determine whether each field is mentioned or not."},
                            {"role": "user", "content": response["description"]}
                        ],
                        response_format=MainInfo
                    )
                    extracted_info = completion.choices[0].message.parsed
                    ret[model].append({
                        "user_id": response["user_id"],
                        "response_i": response["response_i"],
                        "selected": response["selected"],
                        "main_info": response["main_info"],
                        "rating": response["rating"],
                        "extracted_info": extracted_info.dict()
                    })

                    correct, count, mismatched_keys = calculate_accuracy(extracted_info.dict(), response["main_info"])
                metrics[model][0].append(correct)
                metrics[model][1].append(count)
                checked[model].add(response_i)
                for mismatch_item in mismatched_keys:
                    mismatched_counters[mismatch_item["key"]] += 1
                    model_mismatches.append({
                        "description": response["description"],
                        "key": mismatch_item["key"],
                        "extracted_value": str(mismatch_item["extracted_value"]),
                        "original_value": str(mismatch_item["original_value"])
                    })

                # Update the progress bar with the current accuracy
                total_correct = sum(metrics[model][0])
                total_count = sum(metrics[model][1])
                accuracy = total_correct / total_count if total_count > 0 else 0
                response_pbar.set_postfix(accuracy=f"{accuracy:.2%}")

            except Exception as e:
                response_pbar.write(
                    f"Error processing user {response['user_id']} response {response['response_i']}: {e}, extracted_info: {extracted_info.dict()}")
                traceback.print_exc()
                exit()

        pickle.dump([ret, metrics, checked], open(ckpt_filename, "wb"))
        mismatch[model] = mismatched_counters
        write_mismatch_csv(model, model_mismatches)

    # Final accuracy report
    print("\nFinal Accuracy Report:")
    print(mismatch.keys())
    final_metrics = dict()
    for model in metrics:
        total_correct = sum(metrics[model][0])
        total_count = sum(metrics[model][1])
        accuracy = total_correct / total_count if total_count > 0 else 0
        print(f"{model}: {accuracy:.2%}")
        # compute std
        std = np.sqrt(accuracy * (1-accuracy) / total_count)
        print(f"Mismatched keys: {mismatch[model]}")
        print(f"Mismatch CSV file created: {model}_mismatches.csv")
        final_metrics[model] = (accuracy, std)

    # sort the models by accuracy
    sorted_models = sorted(final_metrics.items(), key=lambda x: x[1], reverse=True)
    print("\nSorted Models by Accuracy:")
    for model, _metric in sorted_models:
        accuracy = _metric[0]
        std = _metric[1]
        print(f"{model}: {accuracy:.2%} ({std:.2%})")



