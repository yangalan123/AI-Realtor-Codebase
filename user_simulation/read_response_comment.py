import json
import numpy as np

def replace_model_name(name):
    if name == "none":
        return "GPT4o-mini"
    return name

if __name__ == '__main__':
    users = json.load(open("responses_latest.json", "r", encoding='utf-8'))
    race_dict = {}
    all_models = set()
    for user in users:
        print("-------user:{}-------".format(user["id"]))
        for response in user["responses"]:
            models = response["models"]
            all_models |= set(models)
            selection = response["selection"]
            winner_model = replace_model_name(models[selection])
            loser_model = replace_model_name(models[1 - selection])
            comment = response["reason"]
            rating = response["rating"]
            print("models: {}, user preferred model: {}, comment: {}, rating: {}/5".format(models, models[selection], comment, rating))
            race_key = "{} wins over {}".format(winner_model, loser_model)
            if race_key not in race_dict:
                race_dict[race_key] = {
                    "count": 0,
                    "ratings": [],
                    "comments": [],
                    "users": set()
                }
            race_dict[race_key]["count"] += 1
            race_dict[race_key]["ratings"].append(rating)
            race_dict[race_key]["comments"].append(comment)
            race_dict[race_key]['users'].add(user["id"])
    print("All models: ", all_models)
    for race_key in race_dict:
        print("-------{}-------".format(race_key))
        avg_rating = np.mean(race_dict[race_key]['ratings'])
        print("average rating: ", avg_rating)
        print("race count: ", race_dict[race_key]['count'])
        print("unique users: ", len(race_dict[race_key]['users']))
        print("comments:")
        print("\n".join(race_dict[race_key]['comments']))
