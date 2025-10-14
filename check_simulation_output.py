import copy
import glob

import torch
import json
import math
import numpy as np
from collections import Counter
from pydantic import BaseModel, Field
from tqdm import tqdm

class ScoresForBothDescriptions(BaseModel):
    reasoning_for_description_0: str = Field(...,
                                             description="Please first generate an analysis of the user's profile and history (if available),"
                                                         "and then analyze why the user might prefer the first description."
                                                         "You can use the following format: 'The user might prefer the first description because...'.",
                                             )
    score_for_description_0: float = Field(..., description="The score for the first description", ge=0, le=100)

    reasoning_for_description_1: str = Field(...,
                                             description="Please first generate an analysis of the user's profile and history (if available),"
                                                         "and then analyze why the user might prefer the second description."
                                                         "You can use the following format: 'The user might prefer the second description because...'.",
                                             )
    score_for_description_1: float = Field(..., description="The score for the second description", ge=0, le=100)

class Reflection(BaseModel):
    reflection: str = Field(...,
                            description="Please first look at the user's profile, predicted preferences, actual user selection history and associated reasons (if available),"
                                        "and then reflect what you learn from the user's preferences for all the information so far."
                                        "You can use the following format: 'Based on initial user profile and preference history so far, the reflection on user preference is...'")

def compute_shot_accuracy(accuracy):
    shot_wise_accuracy = dict()
    for user_id in accuracy:
        for shot_i in accuracy[user_id]:
            if shot_i not in shot_wise_accuracy:
                shot_wise_accuracy[shot_i] = []
            shot_wise_accuracy[shot_i].append(accuracy[user_id][shot_i][0] / accuracy[user_id][shot_i][1])
    return shot_wise_accuracy

def analysis_dual_results(dual, minimum_count=3):
    winners = set()
    ties = set()
    for pair, pair_vote in dual.most_common():
        flipped_dual = (pair[1], pair[0])
        flipped_vote = dual[flipped_dual]
        if pair_vote >= minimum_count and flipped_vote >= minimum_count:
            if pair_vote > flipped_vote:
                winner = pair[0]
                loser = pair[1]
                print(f"{winner} wins over {loser} with {pair_vote} > {flipped_vote} votes")
                winners.add((winner, loser))
            elif pair_vote < flipped_vote:
                winner = pair[1]
                loser = pair[0]
                print(f"{winner} wins over {loser} with {flipped_vote} > {pair_vote} votes")
                winners.add((winner, loser))
            else:
                if pair in ties or flipped_dual in ties:
                    continue
                print(f"{pair[0]} ties with {pair[1]} with {pair_vote} votes")
                ties.add(pair)
                ties.add(flipped_dual)
    return winners, ties

def compute_elo_scores_vectorized(win_counts, base_rating=1000, k_factor=10, iterations=100):
    # Extract unique players and create a mapping
    players = sorted(set(player for pair in win_counts.keys() for player in pair))
    player_indices = {player: i for i, player in enumerate(players)}
    n_players = len(players)

    # Initialize ratings array with float64 dtype
    ratings = np.full(n_players, base_rating, dtype=np.float64)

    # Create matrices for wins and games with float64 dtype
    wins_matrix = np.zeros((n_players, n_players), dtype=np.float64)
    games_matrix = np.zeros((n_players, n_players), dtype=np.float64)

    for (player_a, player_b), wins_a in win_counts.items():
        i, j = player_indices[player_a], player_indices[player_b]
        wins_matrix[i, j] = wins_a
        games_matrix[i, j] = wins_a
        games_matrix[j, i] = wins_a

    games_matrix += games_matrix.T

    for _ in range(iterations):
        # Calculate expected scores
        rating_diff = ratings[:, np.newaxis] - ratings
        expected_scores = 1 / (1 + np.power(10, rating_diff / 400))

        # Calculate actual scores
        actual_scores = np.divide(wins_matrix, games_matrix, where=games_matrix!=0)

        # Update ratings
        rating_change = k_factor * (actual_scores - expected_scores)
        ratings += np.sum(rating_change, axis=1)

    return dict(zip(players, ratings))


if __name__ == '__main__':
    root_dir = "/path/to/dir/llm_negotiation/ranking_preference"
    data = f"{root_dir}/responses_latest.json"
    ckpt_filenames = glob.glob(f"{root_dir}/responses_latest/all_responses.pt*gpt-4o-mini.offline")
    # ckpt_filenames = glob.glob(f"{root_dir}/responses_latest_parallel/all_responses.pt*gpt-4o-mini.offline")
    with open(data, "r") as f:
        user_database = json.load(f)
    for ckpt_filename in ckpt_filenames:
        print("processing ", ckpt_filename)

        # ckpt_filename = f"{root_dir}/responses_latest/all_responses.pt.gpt-4o-mini.offline"
        # we check 4-shot results
        # checked_shot = 3
        checkpoint_data = torch.load(ckpt_filename)
        accuracy_profile_filename = ckpt_filename.replace("all_responses", "accuracy")
        accuracy_profile = torch.load(accuracy_profile_filename)
        shot_wise_accuracy = compute_shot_accuracy(accuracy_profile)
        shot_wise_accuracy = {k: np.mean(v) for k, v in shot_wise_accuracy.items()}
        # get the shot with the highest accuracy, the shot num must <= 6
        _max_shot = 0
        _tmp_acc = 0
        for shot, accuracy in shot_wise_accuracy.items():
            if accuracy > _tmp_acc and shot <= 6:
                _max_shot = shot
                _tmp_acc = accuracy
        checked_shot = _max_shot
        output_filename = ckpt_filename + "{}-shot.json".format(checked_shot)
        print("Checked shot: {}, accuracy: {}".format(checked_shot, _tmp_acc))
        real_dual = Counter()
        predicted_dual = Counter()
        all_models = set()
        modified_user_database = []
        id_set = set([user["id"] for user in user_database])
        print("Total user number: ", len(id_set))
        decision_dict = dict()
        for user in tqdm(user_database):
            user_id = user['id']
            responses = user['responses']
            selections = [response['selection'] for response in responses]
            if user_id not in decision_dict:
                decision_dict[user_id] = []
            decision_dict[user_id].append(selections)
        banned_user_ids = set()
        for user_id in decision_dict:
            if len(decision_dict[user_id]) > 1:
                print(f"User {user_id} has more than 1 decision records, they are: {decision_dict[user_id]}")
            # check whether multiple records are consistent
            tmp = decision_dict[user_id][0]
            for record in decision_dict[user_id][1:]:
                if tmp != record:
                    print(f"User {user_id} has inconsistent records: {tmp} vs {record}")
                    banned_user_ids.add(user_id)

        for user in tqdm(user_database):
            user_id = user["id"]
            if user_id in banned_user_ids:
                continue
            responses = user["responses"]
            _new_database = dict()
            for key in user:
                if key != "responses":
                    _new_database[key] = copy.deepcopy(user[key])
            _new_database["responses"] = []
            user_checkpoint = checkpoint_data[user_id][checked_shot]
            preferences, selections, history, reflection = user_checkpoint
            assert len(preferences) == len(responses) - checked_shot, f"prediction length {len(preferences)} does not match remained response length {len(responses) - checked_shot}"
            assert len(selections) == len(responses) - checked_shot, f"selection length {len(selections)} does not match remained response length {len(responses) - checked_shot}"
            for response_i, response in enumerate(responses[checked_shot:]):
                _preference = preferences[response_i]
                prediction = _preference.score_for_description_1 > _preference.score_for_description_0
                selection = selections[response_i]
                _selection_real = response["selection"]
                assert selection == _selection_real, f"selection {selection} does not match response {response['selection']}"
                real_selected_model = response["models"][_selection_real]
                real_loser_model = response["models"][1 - _selection_real]
                predict_selected_model = response["models"][prediction]
                predict_loser_model = response["models"][1 - prediction]
                real_dual[(real_selected_model, real_loser_model)] += 1
                predicted_dual[(predict_selected_model, predict_loser_model)] += 1
                all_models.add(real_selected_model)
                all_models.add(real_loser_model)
                new_response_item = copy.deepcopy(response)
                new_response_item["model_predicted_winner"] = predict_selected_model
                new_response_item["model_predicted_loser"] = predict_loser_model
                _new_database["responses"].append(new_response_item)
            modified_user_database.append(_new_database)

        # save modified user database
        with open(output_filename, "w") as f:
            json.dump(modified_user_database, f, indent=4)
        print("Analysis for real predictions: ")
        winners, tiers = analysis_dual_results(real_dual)
        print("Analysis for predicted predictions: ")
        predict_winners, predict_tiers = analysis_dual_results(predicted_dual)
        print("difference between real and predicted winners: ")
        print(winners - predict_winners)
        print("difference between real and predicted tiers: ")
        print(tiers - predict_tiers)





        print("Real duals:")
        print(real_dual)
        print("Predicted duals:")
        print(predicted_dual)

        real_elo = compute_elo_scores_vectorized(real_dual)
        predicted_elo = compute_elo_scores_vectorized(predicted_dual)
        print("Real ELO scores:")
        print(real_elo)
        print("Predicted ELO scores:")
        print(predicted_elo)

                # assert prediction == response["models"][selection], f"prediction {prediction} does not match response {response['models'][selection]}"
                # print(f"User {user_id} shot {checked_shot + response_i} prediction matches response")
