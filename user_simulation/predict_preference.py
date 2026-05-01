import asyncio
import copy
import json
import os
import os.path
from tqdm.asyncio import tqdm_asyncio
import numpy as np
import torch
from langchain.output_parsers import PydanticOutputParser
from loguru import logger
from pydantic import BaseModel, Field
from sotopia.generation_utils.generate import agenerate
from tqdm import tqdm
import argparse
from asyncio import Lock

NAIVE_FEW_SHOT = "naive_few_shot"
DO_NOT_USE_HISTORY = "do_not_use_history"
DO_NOT_USE_USER_PROFILE = "do_not_use_user_profile"


class Integer(BaseModel):
    reasoning: str = Field(...,
                           description="Please first generate an analysis of the user's profile and history,"
                                       "and then analyze why the user might prefer the first and second descriptions."
                                       "You can use the following format: 'The user might prefer the first description because...'"
                                       "and 'The user might prefer the second description because...'.",
                           )
    value: int = Field(...,
                       description="0 indicates the first description is better, and 1 indicates the second description is better",
                       ge=0, le=1)


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


def processing_listing(listing):
    new_listing = dict()
    for key in listing:
        if key == "listing_id":
            continue
        new_listing[key] = listing[key]
    return new_listing


def generate_prediction_prompt(history, do_not_use_reflection=False, do_not_use_user_profile=False):
    if do_not_use_reflection:
        reflection_line = "{history}\n\n" if history else ""
    else:
        reflection_line = "{history}\n\nReflection: {reflection}\n\n" if history else ""
    # logger.info("Reflection line: {}".format(reflection_line))
    user_profile_line = "User Profile: {user_profile}\n\n" if not do_not_use_user_profile else ""
    return ("You will be given a user profile, a listing and two descriptions of this listing. "
            "Optionally, you may also be given the user's history of preferences. "
            "Your task is to predict which description the user would prefer. \n\n"
            + user_profile_line
            + reflection_line
            + "Listing: {listing}\n\n"
            + "Description 0: {description_0}\n\n"
            + "Description 1: {description_1}\n\n")


def compute_shot_accuracy(accuracy):
    shot_wise_accuracy = dict()
    for user_id in accuracy:
        for shot_i in accuracy[user_id]:
            if shot_i not in shot_wise_accuracy:
                shot_wise_accuracy[shot_i] = []
            shot_wise_accuracy[shot_i].append(accuracy[user_id][shot_i][0] / accuracy[user_id][shot_i][1])
    return shot_wise_accuracy


async def process_existing_responses(user_id, all_responses, accuracy, eval_mode):
    history = []
    reflection = ""
    for num_shot, (preferences, selections, _history, _reflection) in enumerate(all_responses[user_id]):
        if num_shot not in accuracy[user_id]:
            accuracy[user_id][num_shot] = [0, 0]

        if eval_mode == "online":
            judge = int(preferences.score_for_description_1 > preferences.score_for_description_0)
            if judge == selections:
                accuracy[user_id][num_shot][0] += 1
            accuracy[user_id][num_shot][1] += 1
        else:
            for preference, selection in zip(preferences, selections):
                judge = int(preference.score_for_description_1 > preference.score_for_description_0)
                if judge == selection:
                    accuracy[user_id][num_shot][0] += 1
                accuracy[user_id][num_shot][1] += 1

        history = _history
        reflection = _reflection

    return history, reflection


async def process_response(user_id, response, num_shot, model_name, eval_mode, exp_name, all_responses, accuracy,
                           history, reflection, user_profile, history_info):
    listing = processing_listing(response["listing"])
    descriptions = response["descriptions"]
    selection = int(response["selection"])
    user_rating = response["rating"]
    user_reason = response["reason"]

    input_values = {
        "listing": listing,
        "description_0": descriptions[0],
        "description_1": descriptions[1],
    }
    if DO_NOT_USE_USER_PROFILE not in exp_name:
        input_values["user_profile"] = user_profile
    if history and DO_NOT_USE_HISTORY not in exp_name:
        input_values["history"] = history_info
        if NAIVE_FEW_SHOT not in exp_name:
            input_values["reflection"] = reflection

    predicted_preference = await agenerate(
        model_name,
        generate_prediction_prompt(history,
                                   do_not_use_reflection=NAIVE_FEW_SHOT in exp_name or DO_NOT_USE_HISTORY in exp_name,
                                   do_not_use_user_profile=DO_NOT_USE_USER_PROFILE in exp_name),
        input_values=input_values,
        output_parser=PydanticOutputParser(pydantic_object=ScoresForBothDescriptions),
        bad_output_process_model="gpt-4o-mini" if "gpt-4o" in model_name else model_name
    )

    judge = int(predicted_preference.score_for_description_1 > predicted_preference.score_for_description_0)

    if eval_mode == "online":
        all_responses[user_id].append((predicted_preference, selection, history.copy(), reflection))
        if num_shot not in accuracy[user_id]:
            accuracy[user_id][num_shot] = [0, 0]
        if judge == selection:
            accuracy[user_id][num_shot][0] += 1
        accuracy[user_id][num_shot][1] += 1
    else:
        # For offline mode, we'll return the prediction and let the caller handle it
        return predicted_preference, judge

    if DO_NOT_USE_HISTORY not in exp_name:
        history.append(
            f"Listing: {listing}\n\n"
            f"Description 0: {descriptions[0]}\n\n"
            f"Description 1: {descriptions[1]}\n\n"
            f"Predicted Preference: {judge} ({descriptions[judge]}) \n\n"
            f"User Preference: {selection} ({descriptions[selection]})\n\n"
            f"User Rating: {user_rating}\n\n"
            f"User Reason: {user_reason}\n\n"
        )
        history_info = '[Begin of History]\n\n' + '\n\n'.join(history) + '\n\n[End of History]'
        if NAIVE_FEW_SHOT not in exp_name:
            reflection = await agenerate(
                model_name,
                "Based on the user's profile, predicted preferences, actual user selection history and associated reasons so far,"
                "please provide reflection on user preference. \n\n"
                "History-so-far: \n{history}\n",
                input_values={"history": history_info},
                output_parser=PydanticOutputParser(pydantic_object=Reflection),
                bad_output_process_model="gpt-4o-mini" if "gpt-4o" in model_name else model_name
            )

    return history, reflection


async def process_user(user, model_name, eval_mode, force_recompute, stop_shot, exp_name, all_responses, accuracy,
                       user_locks):
    user_id = user['id']
    user_profile = user["profile"]

    if user_id not in accuracy:
        accuracy[user_id] = dict()

    if user_id not in user_locks:
        user_locks[user_id] = Lock()

    async with user_locks[user_id]:
        if user_id in all_responses and not force_recompute:
            history, reflection = await process_existing_responses(user_id, all_responses, accuracy, eval_mode)
        else:
            all_responses[user_id] = []
            history = []
            reflection = ""

        history_info = '[Begin of History]\n\n' + '\n\n'.join(history) + '\n\n[End of History]' if history else ""

        if eval_mode == "online":
            for response_i, response in enumerate(user["responses"][len(all_responses[user_id]):]):
                num_shot = len(all_responses[user_id]) + response_i
                if num_shot > stop_shot:
                    break

                history, reflection = await process_response(user_id, response, num_shot, model_name, eval_mode,
                                                             exp_name, all_responses, accuracy, history, reflection,
                                                             user_profile, history_info)
        else:
            for num_shot, responses in enumerate(user["responses"][len(all_responses[user_id]):]):
                if num_shot > stop_shot:
                    break

                predictions = []
                selections = []
                for response in responses:
                    predicted_preference, judge = await process_response(user_id, response, num_shot, model_name,
                                                                         eval_mode, exp_name, all_responses, accuracy,
                                                                         history, reflection, user_profile,
                                                                         history_info)
                    predictions.append(predicted_preference)
                    selections.append(int(response["selection"]))

                all_responses[user_id].append((predictions, selections, history.copy(), reflection))

                if num_shot not in accuracy[user_id]:
                    accuracy[user_id][num_shot] = [0, 0]

                for pred, sel in zip(predictions, selections):
                    judge = int(pred.score_for_description_1 > pred.score_for_description_0)
                    if judge == sel:
                        accuracy[user_id][num_shot][0] += 1
                    accuracy[user_id][num_shot][1] += 1

                # Update history and reflection after processing all responses for this shot
                if DO_NOT_USE_HISTORY not in exp_name:
                    for response, pred in zip(responses, predictions):
                        history.append(
                            f"Listing: {processing_listing(response['listing'])}\n\n"
                            f"Description 0: {response['descriptions'][0]}\n\n"
                            f"Description 1: {response['descriptions'][1]}\n\n"
                            f"Predicted Preference: {int(pred.score_for_description_1 > pred.score_for_description_0)} "
                            f"({response['descriptions'][int(pred.score_for_description_1 > pred.score_for_description_0)]}) \n\n"
                            f"User Preference: {response['selection']} ({response['descriptions'][response['selection']]})\n\n"
                            f"User Rating: {response['rating']}\n\n"
                            f"User Reason: {response['reason']}\n\n"
                        )
                    history_info = '[Begin of History]\n\n' + '\n\n'.join(history) + '\n\n[End of History]'
                    if NAIVE_FEW_SHOT not in exp_name:
                        reflection = await agenerate(
                            model_name,
                            "Based on the user's profile, predicted preferences, actual user selection history and associated reasons so far,"
                            "please provide reflection on user preference. \n\n"
                            "History-so-far: \n{history}\n",
                            input_values={"history": history_info},
                            output_parser=PydanticOutputParser(pydantic_object=Reflection),
                            bad_output_process_model="gpt-4o-mini" if "gpt-4o" in model_name else model_name
                        )

    return user_id, all_responses[user_id], accuracy[user_id]


async def process_existing_responses(user_id, all_responses, accuracy, eval_mode):
    history = []
    reflection = ""
    for num_shot, (preferences, selections, _history, _reflection) in enumerate(all_responses[user_id]):
        if num_shot not in accuracy[user_id]:
            accuracy[user_id][num_shot] = [0, 0]

        if eval_mode == "online":
            judge = int(preferences.score_for_description_1 > preferences.score_for_description_0)
            if judge == selections:
                accuracy[user_id][num_shot][0] += 1
            accuracy[user_id][num_shot][1] += 1
        else:
            for preference, selection in zip(preferences, selections):
                judge = int(preference.score_for_description_1 > preference.score_for_description_0)
                if judge == selection:
                    accuracy[user_id][num_shot][0] += 1
                accuracy[user_id][num_shot][1] += 1

        history = _history
        reflection = _reflection

    return history, reflection


async def process_response(user_id, response, num_shot, model_name, eval_mode, exp_name, all_responses, accuracy,
                           history, reflection, user_profile, history_info):
    listing = processing_listing(response["listing"])
    descriptions = response["descriptions"]
    selection = int(response["selection"])
    user_rating = response["rating"]
    user_reason = response["reason"]

    input_values = {
        "listing": listing,
        "description_0": descriptions[0],
        "description_1": descriptions[1],
    }
    if DO_NOT_USE_USER_PROFILE not in exp_name:
        input_values["user_profile"] = user_profile
    if history and DO_NOT_USE_HISTORY not in exp_name:
        input_values["history"] = history_info
        if NAIVE_FEW_SHOT not in exp_name:
            input_values["reflection"] = reflection

    predicted_preference = await agenerate(
        model_name,
        generate_prediction_prompt(history,
                                   do_not_use_reflection=NAIVE_FEW_SHOT in exp_name or DO_NOT_USE_HISTORY in exp_name,
                                   do_not_use_user_profile=DO_NOT_USE_USER_PROFILE in exp_name),
        input_values=input_values,
        output_parser=PydanticOutputParser(pydantic_object=ScoresForBothDescriptions),
        bad_output_process_model="gpt-4o-mini" if "gpt-4o" in model_name else model_name
    )

    judge = int(predicted_preference.score_for_description_1 > predicted_preference.score_for_description_0)

    if eval_mode == "online":
        all_responses[user_id].append((predicted_preference, selection, history.copy(), reflection))
        if num_shot not in accuracy[user_id]:
            accuracy[user_id][num_shot] = [0, 0]
        if judge == selection:
            accuracy[user_id][num_shot][0] += 1
        accuracy[user_id][num_shot][1] += 1
    else:
        # For offline mode, we'll return the prediction and let the caller handle it
        return predicted_preference, judge

    if DO_NOT_USE_HISTORY not in exp_name:
        history.append(
            f"Listing: {listing}\n\n"
            f"Description 0: {descriptions[0]}\n\n"
            f"Description 1: {descriptions[1]}\n\n"
            f"Predicted Preference: {judge} ({descriptions[judge]}) \n\n"
            f"User Preference: {selection} ({descriptions[selection]})\n\n"
            f"User Rating: {user_rating}\n\n"
            f"User Reason: {user_reason}\n\n"
        )
        history_info = '[Begin of History]\n\n' + '\n\n'.join(history) + '\n\n[End of History]'
        if NAIVE_FEW_SHOT not in exp_name:
            reflection = await agenerate(
                model_name,
                "Based on the user's profile, predicted preferences, actual user selection history and associated reasons so far,"
                "please provide reflection on user preference. \n\n"
                "History-so-far: \n{history}\n",
                input_values={"history": history_info},
                output_parser=PydanticOutputParser(pydantic_object=Reflection),
                bad_output_process_model="gpt-4o-mini" if "gpt-4o" in model_name else model_name
            )

    return history, reflection


async def process_user(user, model_name, eval_mode, force_recompute, stop_shot, exp_name, all_responses, accuracy,
                       user_locks):
    user_id = user['id']
    user_profile = user["profile"]

    if user_id not in accuracy:
        accuracy[user_id] = dict()

    if user_id not in user_locks:
        user_locks[user_id] = Lock()

    # async with user_locks[user_id]:
    if user_id in all_responses and not force_recompute:
        history, reflection = await process_existing_responses(user_id, all_responses, accuracy, eval_mode)
    else:
        all_responses[user_id] = []
        history = []
        reflection = ""

    history_info = '[Begin of History]\n\n' + '\n\n'.join(history) + '\n\n[End of History]' if history else ""
    saved_existing_responses = len(all_responses[user_id])

    if eval_mode == "online":
        for response_i, response in enumerate(user["responses"][len(all_responses[user_id]):]):
            num_shot = saved_existing_responses + response_i
            if num_shot > stop_shot:
                break

            history, reflection = await process_response(user_id, response, num_shot, model_name, eval_mode,
                                                         exp_name, all_responses, accuracy, history, reflection,
                                                         user_profile, history_info)
            history_info = '[Begin of History]\n\n' + '\n\n'.join(history) + '\n\n[End of History]'
    else:
        for response_i, responses in enumerate(user["responses"][saved_existing_responses:]):
            num_shot = saved_existing_responses + response_i
            if num_shot > stop_shot:
                break

            predictions = dict()
            selections = dict()
            for offline_response_i, offline_response in enumerate(user["responses"][num_shot: ]):
                predicted_preference, judge = await process_response(user_id, offline_response, num_shot, model_name,
                                                                 eval_mode, exp_name, all_responses, accuracy,
                                                                 history, reflection, user_profile,
                                                                 history_info)
                # predictions.append(predicted_preference)
                # selections.append(int(offline_response["selection"]))
                predictions[offline_response_i] = predicted_preference
                selections[offline_response_i] = int(offline_response["selection"])
            # convert predictions and selections to list
            predictions = [predictions[i] for i in range(len(predictions))]
            selections = [selections[i] for i in range(len(selections))]

            all_responses[user_id].append((predictions, selections, history.copy(), reflection))

            if num_shot not in accuracy[user_id]:
                accuracy[user_id][num_shot] = [0, 0]

            for pred, sel in zip(predictions, selections):
                judge = int(pred.score_for_description_1 > pred.score_for_description_0)
                if judge == sel:
                    accuracy[user_id][num_shot][0] += 1
                accuracy[user_id][num_shot][1] += 1

            # Update history and reflection after processing all responses for this shot
            if DO_NOT_USE_HISTORY not in exp_name:
                for response, pred in zip(responses, predictions):
                    history.append(
                        f"Listing: {processing_listing(response['listing'])}\n\n"
                        f"Description 0: {response['descriptions'][0]}\n\n"
                        f"Description 1: {response['descriptions'][1]}\n\n"
                        f"Predicted Preference: {int(pred.score_for_description_1 > pred.score_for_description_0)} "
                        f"({response['descriptions'][int(pred.score_for_description_1 > pred.score_for_description_0)]}) \n\n"
                        f"User Preference: {response['selection']} ({response['descriptions'][response['selection']]})\n\n"
                        f"User Rating: {response['rating']}\n\n"
                        f"User Reason: {response['reason']}\n\n"
                    )
                history_info = '[Begin of History]\n\n' + '\n\n'.join(history) + '\n\n[End of History]'
                if NAIVE_FEW_SHOT not in exp_name:
                    reflection = await agenerate(
                        model_name,
                        "Based on the user's profile, predicted preferences, actual user selection history and associated reasons so far,"
                        "please provide reflection on user preference. \n\n"
                        "History-so-far: \n{history}\n",
                        input_values={"history": history_info},
                        output_parser=PydanticOutputParser(pydantic_object=Reflection),
                        bad_output_process_model="gpt-4o-mini" if "gpt-4o" in model_name else model_name
                    )

    return user_id, all_responses[user_id], accuracy[user_id]


async def process_user_with_error_handling(user, model_name, eval_mode, force_recompute, stop_shot, exp_name,
                                           all_responses, accuracy, user_locks):
    user_id = user['id']
    try:
        return await process_user(user, model_name, eval_mode, force_recompute, stop_shot, exp_name, all_responses,
                                  accuracy, user_locks)
    except Exception as e:
        logger.error(f"Error processing user {user_id}: {str(e)}")
        # print stack trace
        import traceback
        traceback.print_exc()
        return user_id, None, None


async def save_checkpoint(root_dir, suffix, all_responses, accuracy):
    ckpt_name = os.path.join(root_dir, f"all_responses_checkpoint.pt{suffix}")
    accuracy_name = os.path.join(root_dir, f"accuracy_checkpoint.pt{suffix}")
    torch.save(all_responses, ckpt_name)
    torch.save(accuracy, accuracy_name)


async def main(args):
    users = json.load(open(args.data, "r", encoding='utf-8'))
    root_dir = args.data.split(".")[0] + "_parallel"
    os.makedirs(root_dir, exist_ok=True)

    model_name = args.model_name
    exp_name = args.exp_name.replace("|", ".")
    eval_mode = args.eval_mode
    force_recompute = args.force_recompute
    stop_shot = args.stop_shot

    if DO_NOT_USE_HISTORY in exp_name:
        logger.info("Do not use history, so we set stop shot to 0")
        stop_shot = 0

    suffix = f".{exp_name}.{'llama3' if 'custom' in model_name else model_name}.{eval_mode}"
    logger.add(os.path.join(root_dir, f"predict_preference.log{suffix}"), rotation="10 MB")

    ckpt_name = os.path.join(root_dir, f"all_responses.pt{suffix}")
    all_responses = torch.load(ckpt_name) if os.path.exists(ckpt_name) else dict()
    accuracy = dict()
    user_locks = {}

    logger.info(f"Start predicting preferences, number of users: {len(users)}")

    async def process_users():
        tasks = [
            process_user_with_error_handling(user, model_name, eval_mode, force_recompute, stop_shot, exp_name,
                                             all_responses, accuracy, user_locks)
            for user in users
        ]

        checkpoint_interval = max(1, len(users) // 10)  # Save checkpoint every 10% of users
        processed_users = 0

        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing users"):
            user_id, user_responses, user_accuracy = await task
            if user_responses is not None and user_accuracy is not None:
                all_responses[user_id] = user_responses
                accuracy[user_id] = user_accuracy

            processed_users += 1
            if processed_users % checkpoint_interval == 0:
                await save_checkpoint(root_dir, suffix, all_responses, accuracy)
                logger.info(f"Saved checkpoint after processing {processed_users} users")

    await process_users()

    # Save final results
    torch.save(all_responses, ckpt_name)
    torch.save(accuracy, os.path.join(root_dir, f"accuracy.pt{suffix}"))

    log_statistics(accuracy)

def log_statistics(accuracy):
    user_accuracy = {k: sum(vv[0] for vv in v.values()) / sum(vv[1] for vv in v.values()) for k, v in accuracy.items()}
    logger.info(f"User-wise accuracy: {user_accuracy}")
    user_std = np.std(list(user_accuracy.values()))
    logger.info(f"User-wise std: {user_std}")

    shot_wise_accuracy = compute_shot_accuracy(accuracy)
    shot_wise_accuracy = {k: np.mean(v) for k, v in shot_wise_accuracy.items()}
    logger.info(f"Shot-wise accuracy: {shot_wise_accuracy}")
    shot_wise_std = {k: np.std(v) for k, v in shot_wise_accuracy.items()}
    logger.info(f"Shot-wise std: {shot_wise_std}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PredictPreferenceArgsParsing.')
    parser.add_argument("--model_name", type=str,
                        default="custom/meta-llama/Meta-Llama-3.1-70B-Instruct@http://localhost:8000/v1", )
    parser.add_argument("--eval_mode", type=str, default="online", help="online or offline")
    parser.add_argument("--data", type=str, default="collected_human_data.json", help="data file")
    parser.add_argument("--force_recompute", action="store_true", help="force recompute")
    parser.add_argument("--stop_shot", type=int, default=99, help="stop shot")
    parser.add_argument("--exp_name", type=str, default="",
                        help="experiment name, by default we will use history 4-shot and reflection")
    args = parser.parse_args()
    asyncio.run(main(args))
