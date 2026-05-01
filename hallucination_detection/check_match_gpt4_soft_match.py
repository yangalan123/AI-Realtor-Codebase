import os
import json
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm
import pickle
import traceback
import scipy.stats as st


class MainInfo(BaseModel):
    # price_mentioned: bool
    # price: float
    address_mentioned: bool
    address: str
    home_insights_mentioned: bool
    home_insights: list[str]
    # living_area_mentioned: bool
    # living_area: str
    # bedrooms_mentioned: bool
    # bedrooms: float
    # bathrooms_mentioned: bool
    # bathrooms: float


def check_feature_accuracy(client, description, true_value, extracted_value, feature_name):
    prompt = f"""
Given the following information:

1. Description: {description}
2. True value for {feature_name}: {json.dumps(true_value)}
3. Extracted value for {feature_name}: {json.dumps(extracted_value)}

Please analyze how well the extracted value matches the true value, considering the context provided in the description.

For 'home_insights', consider it a good match if a significant subset of the true insights is correctly identified.
For 'address', consider it a good match if at least a subset (e.g., city/state) is correctly identified, given it was mentioned in the description.

Provide a score between 0 and 10, where:
0 = Completely incorrect or irrelevant
5 = Partially correct or relevant
10 = Perfect match

Respond with a JSON object in the following format:
{{
    "score": int
}}

Where 'score' is an integer between 0 and 10.
"""

    completion = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that analyzes real estate information."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
    )

    result = json.loads(completion.choices[0].message.content)
    return result["score"]


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
    ckpt_filename = "extracted_info_gpt4_score_soft_match_v2.pkl"
    if os.path.exists(ckpt_filename):
        ret, metrics, checked = pickle.load(open(ckpt_filename, "rb"))

    model_pbar = tqdm(model_wise_responses.items(), desc="Processing models", total=len(model_wise_responses),
                      position=0)
    for model, responses in model_pbar:
        model_pbar.set_description(f"Processing model {model}")
        if model not in checked:
            ret[model] = []
            metrics[model] = []
            checked[model] = set()

        response_pbar = tqdm(enumerate(responses), total=len(responses), desc=f"Processing responses", position=1,
                             leave=False)
        for response_i, response in response_pbar:
            try:
                if response_i in checked[model]:
                    accuracy_result = ret[model][response_i]["accuracy_result"]
                else:
                    example_home_insights =["Large island", "Oversized bathroom", "Open floor plan", "Lake views", "Orange l lines", "Newer stainless steel appliances", "Gorgeous hardwood floors", "Tons of cabinet space", "In-unit washer and dryer", "Skyline view", "Private balcony", "Beautiful city"]
                    example_addr = "1255 S State St UNIT 703 Chicago IL 60601"
                    completion = client.beta.chat.completions.parse(
                        model="gpt-4o-mini-2024-07-18",
                        messages=[
                            {"role": "system",
                             "content": "Extract Real Estate Information. Find the home insights (e.g., {}), and address (e.g., {}) from the description. Not all information may be present, so you also have to determine whether each field is mentioned or not.".format(example_home_insights, example_addr)},
                            {"role": "user", "content": response['description']}
                        ],
                        response_format=MainInfo
                    )
                    extracted_info = completion.choices[0].message.parsed

                    accuracy_result = {}
                    total_score = 0
                    max_possible_score = 0
                    for feature in ["home_insights", "address"]:
                    #for feature in ["price", "home_insights", "living_area", "bedrooms", "bathrooms", "address"]:
                        if getattr(extracted_info, f"{feature}_mentioned"):
                            score = check_feature_accuracy(
                                client,
                                response["description"],
                                response["main_info"][feature],
                                getattr(extracted_info, feature),
                                feature
                            )
                            accuracy_result[f"{feature}_score"] = score
                            total_score += score
                            max_possible_score += 10
                        else:
                            accuracy_result[f"{feature}_score"] = None

                    accuracy_result["total_score"] = total_score
                    accuracy_result["max_possible_score"] = max_possible_score
                    accuracy_result[
                        "normalized_score"] = total_score / max_possible_score if max_possible_score > 0 else 0

                    ret[model].append({
                        "user_id": response["user_id"],
                        "response_i": response["response_i"],
                        "selected": response["selected"],
                        "main_info": response["main_info"],
                        "rating": response["rating"],
                        "extracted_info": extracted_info.dict(),
                        "accuracy_result": accuracy_result
                    })

                metrics[model].append(accuracy_result)
                checked[model].add(response_i)

                # Update the progress bar with the current average normalized score
                avg_normalized_score = sum(r["normalized_score"] for r in metrics[model]) / len(metrics[model])
                response_pbar.set_postfix(avg_score=f"{avg_normalized_score:.2%}")

            except Exception as e:
                response_pbar.write(
                    f"Error processing user {response['user_id']} response {response['response_i']}: {e}")
                traceback.print_exc()
                exit()

        pickle.dump([ret, metrics, checked], open(ckpt_filename, "wb"))

    # Final score report
    print("\nFinal Score Report:")
    for model in metrics:
        avg_normalized_score = sum(r["normalized_score"] for r in metrics[model]) / len(metrics[model])
        sem = st.sem([r["normalized_score"] for r in metrics[model]])
        print(f"{model}: {avg_normalized_score:.2%} ({sem:.2%})")

    # Rank models
    ranked_models = sorted(metrics.keys(),
                           key=lambda m: sum(r["normalized_score"] for r in metrics[m]) / len(metrics[m]), reverse=True)
    print("\nModel Rankings:")
    for rank, model in enumerate(ranked_models, 1):
        avg_normalized_score = sum(r["normalized_score"] for r in metrics[model]) / len(metrics[model])
        print(f"{rank}. {model}: {avg_normalized_score:.2%}")