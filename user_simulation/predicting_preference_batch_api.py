import os
import os.path
import json
import numpy as np
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from openai import OpenAI
from collections import Counter
from pydantic import BaseModel, Field
from tqdm import tqdm
import traceback
import torch
import argparse
import pickle
from loguru import logger
import matplotlib.pyplot as plt

NAIVE_FEW_SHOT = "naive_few_shot"
DO_NOT_USE_HISTORY = "do_not_use_history"
DO_NOT_USE_USER_PROFILE = "do_not_use_user_profile"


class ScoresForBothDescriptions(BaseModel):
    reasoning_for_description_0: str = Field(...,
                                             description="Please first generate an analysis of the user's profile and history (if available),"
                                                         "and then analyze why the user might prefer the first description."
                                                         "You can use the following format: 'The user might prefer the first description because...'.")
    score_for_description_0: float = Field(..., description="The score for the first description", ge=0, le=100)
    reasoning_for_description_1: str = Field(...,
                                             description="Please first generate an analysis of the user's profile and history (if available),"
                                                         "and then analyze why the user might prefer the second description."
                                                         "You can use the following format: 'The user might prefer the second description because...'.")
    score_for_description_1: float = Field(..., description="The score for the second description", ge=0, le=100)


@dataclass
class BatchTask:
    user_id: str
    shot_num: int
    test_instance_id: int
    task_id: str
    description_0: str
    description_1: str
    shuffled_order: bool
    listing: Dict
    user_profile: str
    history: str = ""
    selection: int = 0


class MiniBatchInfo(BaseModel):
    batch_id: int
    batch_job_id: str
    input_file_path: str
    output_file_id: Optional[str] = None
    status: str = "pending"

class BatchCheckpoint(BaseModel):
    total_batches: int
    batch_size: int
    completed_batches: List[int] = []
    mini_batches: Dict[int, MiniBatchInfo] = {}

class PreferencePredictor:
    def __init__(self, model_name: str, output_dir: str, batch_size: int = 100):
        self.model_name = model_name
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.tasks: List[BatchTask] = []
        self.client = OpenAI()
        self.checkpoint_path = os.path.join(output_dir, "batch_checkpoint.json")
        self.registry_path = os.path.join(output_dir, "id_registry.pkl")
        self.used_ids: Dict[str, int] = {}
        self.id_to_data: Dict[str, Dict] = {}
        self.answer_sheet: Dict[int, int] = {}
        self.load_registry()


    def load_registry(self):
        """Load existing ID registry if it exists"""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'rb') as f:
                    # data = json.load(f)
                    # self.used_ids = data['used_ids']
                    # self.id_to_data = data['id_to_data']
                    data = pickle.load(f)
                    self.used_ids = data['used_ids']
                    self.id_to_data = data['id_to_data']
                    self.tasks = data['tasks']
                    self.answer_sheet = data['answer_sheet']
                    logger.info("Loaded existing ID registry, len(used_ids): %d, len(id_to_data): %d, len(tasks): %d" % (
                        len(self.used_ids), len(self.id_to_data), len(self.tasks)))
            except Exception as e:
                logger.error(f"Error loading registry: {e}")
                self.used_ids = {}
                self.id_to_data = {}

    def save_registry(self):
        """Save current ID registry"""
        logger.info("Saving ID registry...already added {} tasks".format(len(self.tasks)))
        pickle.dump({
            'used_ids': self.used_ids,
            'id_to_data': self.id_to_data,
            'tasks': self.tasks,
            'answer_sheet': self.answer_sheet
        }, open(self.registry_path, 'wb'))
        # with open(self.registry_path, 'w') as f:
        #     json.dump({
        #         'used_ids': self.used_ids,
        #         'id_to_data': self.id_to_data
        #     }, f, indent=2)

    def clear_registry(self):
        """Clear current ID registry"""
        self.used_ids = {}
        self.id_to_data = {}
        self.tasks = []
        self.answer_sheet = {}

    def generate_unique_id(self, user_id: str, shot_num: int, listing_id: str,
                           is_shuffled: bool, original_data_index: int, test_instance_id: int) -> str:
        """Generate a unique ID and maintain registry"""
        base_id = f"{user_id}-{shot_num}-{test_instance_id}-{listing_id}-{'shuffled' if is_shuffled else 'normal'}"

        if base_id in self.used_ids:
            self.used_ids[base_id] += 1
            unique_id = f"{base_id}-v{self.used_ids[base_id]}"
        else:
            self.used_ids[base_id] = 0
            unique_id = base_id

        # Store mapping to original data
        self.id_to_data[unique_id] = {
            'user_id': user_id,
            'shot_num': shot_num,
            'listing_id': listing_id,
            'is_shuffled': is_shuffled,
            'original_data_index': original_data_index,
            'test_instance_id': test_instance_id
        }

        # Save registry after each update
        # self.save_registry()
        return unique_id

    def add_task(self, user_id: str, shot_num: int, listing: Dict,
                 description_0: str, description_1: str, user_profile: str,
                 original_data_index: int, test_instance_id: int, selection: int, history: str = "") -> None:
        """Adds two tasks with shuffled description orders"""
        listing_id = listing.get('listing_id', 'unknown')
        # assert original_data_index not in self.answer_sheet, f"Task already exists for index {original_data_index}"
        if original_data_index in self.answer_sheet:
            assert self.answer_sheet[original_data_index] == selection, f"Selection mismatch for index {original_data_index}: {self.answer_sheet[original_data_index]} vs {selection}"
        self.answer_sheet[original_data_index] = selection

        for shuffled in [False, True]:
            task_id = self.generate_unique_id(
                user_id=user_id,
                shot_num=shot_num,
                test_instance_id=test_instance_id,
                listing_id=listing_id,
                is_shuffled=shuffled,
                original_data_index=original_data_index
            )

            task = BatchTask(
                user_id=user_id,
                shot_num=shot_num,
                test_instance_id=test_instance_id,
                task_id=task_id,
                description_0=description_0,
                description_1=description_1,
                shuffled_order=shuffled,
                listing=listing,
                user_profile=user_profile,
                selection=selection if not shuffled else 1 - selection,
                history=history
            )
            self.tasks.append(task)

    def process_results(self, results_file: str) -> Dict[str, Dict[int, Dict]]:
        """Process results from the batch job"""
        user_scores = {}

        with open(results_file, 'r') as f:
            for line in f:
                result = json.loads(line)
                task_id = result['custom_id']

                # Get original data mapping
                if task_id not in self.id_to_data:
                    logger.error(f"Unknown task_id in results: {task_id}")
                    logger.error(f"Task ID counts: {len(self.id_to_data)}")
                    exit()
                    continue

                data = self.id_to_data[task_id]
                user_id = data['user_id']
                shot_num = data['shot_num']
                listing_id = data['listing_id']
                is_shuffled = data['is_shuffled']
                original_index = data['original_data_index']
                test_instance_id = data['test_instance_id']

                # Parse response into Pydantic model
                try:
                    response_content = json.loads(result['response']['body']['choices'][0]['message']['content'])
                    scores = ScoresForBothDescriptions(**response_content)
                except Exception as e:
                    traceback.print_exc()
                    logger.error(f"Error parsing response for task {task_id}: {e}, result: {result}, have to skip (perhaps due to gpt-stop-by-length)")
                    # exit()
                    continue

                # Adjust scores based on shuffling
                # logger.info(scores)
                # exit()
                if is_shuffled:
                    scores = ScoresForBothDescriptions(
                        reasoning_for_description_0=scores.reasoning_for_description_1,
                        score_for_description_0=scores.score_for_description_1,
                        reasoning_for_description_1=scores.reasoning_for_description_0,
                        score_for_description_1=scores.score_for_description_0
                    )

                if user_id not in user_scores:
                    user_scores[user_id] = {}
                if shot_num not in user_scores[user_id]:
                    user_scores[user_id][shot_num] = {}
                if original_index not in user_scores[user_id][shot_num]:
                    user_scores[user_id][shot_num][original_index] = []

                user_scores[user_id][shot_num][original_index].append(scores)

        # Average scores for each user, shot, and original data index
        final_scores = {}
        for user_id, shot_scores in user_scores.items():
            final_scores[user_id] = {}
            for shot_num, index_scores in shot_scores.items():
                final_scores[user_id][shot_num] = {}
                for original_index, predictions in index_scores.items():
                    avg_score_0 = np.mean([p.score_for_description_0 for p in predictions])
                    avg_score_1 = np.mean([p.score_for_description_1 for p in predictions])
                    final_scores[user_id][shot_num][original_index] = ScoresForBothDescriptions(
                        reasoning_for_description_0=predictions[0].reasoning_for_description_0,
                        score_for_description_0=avg_score_0,
                        reasoning_for_description_1=predictions[0].reasoning_for_description_1,
                        score_for_description_1=avg_score_1
                    )
        # logger.info(final_scores[user_id])
        # exit()

        return final_scores

    def save_checkpoint(self, checkpoint: BatchCheckpoint) -> None:
        """Save batch checkpoint information"""
        with open(self.checkpoint_path, 'w') as f:
            f.write(checkpoint.json())

    def load_checkpoint(self) -> Optional[BatchCheckpoint]:
        """Load batch checkpoint information"""
        if os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, 'r') as f:
                return BatchCheckpoint.parse_raw(f.read())
        return None

    def create_prompt(self, task: BatchTask) -> Dict[str, Any]:
        """Creates a prompt for a single task"""
        system_prompt = (
            "You will be given a user profile, a listing and two descriptions of this listing. "
            "Your task is to predict which description the user would prefer. "
            "For each description, provide a reasoning based on the user profile and a score between 0 and 100. "
            "Higher scores indicate stronger preference. Output must follow the exact schema specified. "
            "Scores must be numbers between 0 and 100."
        # Moving constraints to prompt text since we can't use schema constraints
        )

        # Format the user message with proper order of descriptions
        desc_0, desc_1 = (task.description_1, task.description_0) if task.shuffled_order else (
            task.description_0, task.description_1)

        user_message = f"User Profile: {task.user_profile}\n\n"
        if task.history:
            user_message += f"History: {task.history}\n\n"
        user_message += f"Listing: {json.dumps(task.listing)}\n\n"
        user_message += f"Description 0: {desc_0}\n\nDescription 1: {desc_1}"

        return {
            "custom_id": task.task_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.model_name,
                "temperature": 0.7,
                "max_tokens": 500,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "ScoresForBothDescriptions",
                        "strict": True,
                        # "type": "object",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "reasoning_for_description_0": {
                                    "type": "string",
                                    "description": "Analysis of why the user might prefer the first description"
                                },
                                "score_for_description_0": {
                                    "type": "number",
                                    "description": "Score for the first description (between 0 and 100)"
                                },
                                "reasoning_for_description_1": {
                                    "type": "string",
                                    "description": "Analysis of why the user might prefer the second description"
                                },
                                "score_for_description_1": {
                                    "type": "number",
                                    "description": "Score for the second description (between 0 and 100)"
                                }
                            },
                            "required": [
                                "reasoning_for_description_0",
                                "score_for_description_0",
                                "reasoning_for_description_1",
                                "score_for_description_1"
                            ],
                            "additionalProperties": False
                        },
                    }
                }
            }
        }

    async def get_or_create_batch(self, batch_id: int, batch_tasks: List[BatchTask]) -> MiniBatchInfo:
        """Get existing batch or create a new one for a subset of tasks"""
        batch_dir = os.path.join(self.output_dir, f"batch_{batch_id}")
        os.makedirs(batch_dir, exist_ok=True)
        batch_file = os.path.join(batch_dir, "batch_tasks.jsonl")

        # Create new batch file
        with open(batch_file, 'w') as f:
            for task in batch_tasks:
                prompt = self.create_prompt(task)
                f.write(json.dumps(prompt) + '\n')

        logger.info(f"Created batch file: {batch_file}, containing {len(batch_tasks)} tasks")

        batch_file_obj = self.client.files.create(
            file=open(batch_file, "rb"),
            purpose="batch"
        )

        batch_job = self.client.batches.create(
            input_file_id=batch_file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )

        return MiniBatchInfo(
            batch_id=batch_id,
            batch_job_id=batch_job.id,
            input_file_path=batch_file,
            status="pending"
        )

    def get_batch_results(self, batch_info: MiniBatchInfo) -> Optional[str]:
        """Get results for a completed batch"""
        if batch_info.status != "completed":
            return None

        try:
            result = self.client.files.content(batch_info.output_file_id).content
            result_file_path = os.path.join(
                os.path.dirname(batch_info.input_file_path),
                "batch_results.jsonl"
            )
            if os.path.exists(result_file_path):
                # no need to download this file again
                logger.info(f"Results file already exists: {result_file_path}, will load from disk")
                return result_file_path
            else:
                with open(result_file_path, 'wb') as file:
                    file.write(result)
                logger.info(f"Results file downloaded: {result_file_path}")
                return result_file_path
        except Exception as e:
            logger.error(f"Error retrieving results for batch {batch_info.batch_id}: {e} ({batch_info})")
            return None

    async def process_batches(self) -> Optional[Dict]:
        """Process tasks in batches using checkpoints"""
        # Calculate total number of batches
        total_batches = (len(self.tasks) + self.batch_size - 1) // self.batch_size

        # Load or create checkpoint
        checkpoint = self.load_checkpoint()
        logger.info(f"Checkpoint Loaded: {checkpoint}")
        if checkpoint is None:
            checkpoint = BatchCheckpoint(
                total_batches=total_batches,
                batch_size=self.batch_size
            )
        checkpoint.completed_batches.clear()

        # Check status of existing batches and create new ones if needed
        all_results = {}
        logger.info(f"Processing {total_batches} batches...")

        for batch_id in range(total_batches):
            batch_start = batch_id * self.batch_size
            batch_end = min(batch_start + self.batch_size, len(self.tasks))
            batch_tasks = self.tasks[batch_start:batch_end]

            # Skip if batch is already completed
            if batch_id in checkpoint.completed_batches:
                continue

            # Get or create batch job
            if batch_id not in checkpoint.mini_batches:
                batch_info = await self.get_or_create_batch(batch_id, batch_tasks)
                checkpoint.mini_batches[batch_id] = batch_info
                self.save_checkpoint(checkpoint)
            else:
                batch_info = checkpoint.mini_batches[batch_id]

            # Check batch status
            try:
                batch_job = self.client.batches.retrieve(batch_info.batch_job_id)
                logger.info(f"Batch Job at batch_id: {batch_id} is: {batch_job}")
                batch_info.status = batch_job.status

                if batch_job.status == "completed":
                    batch_info.output_file_id = batch_job.output_file_id
                    checkpoint.mini_batches[batch_id] = batch_info
                    logger.info(f"Batch {batch_id} completed")

                    # Process results
                    results_file = self.get_batch_results(batch_info)
                    logger.info(f"Results file: {results_file}")
                    if results_file:
                        batch_results = self.process_results(results_file)

                        # Merge results
                        for user_id, user_data in batch_results.items():
                            if user_id not in all_results:
                                all_results[user_id] = {}
                            for shot_num, shot_data in user_data.items():
                                if shot_num not in all_results[user_id]:
                                    all_results[user_id][shot_num] = {}
                                all_results[user_id][shot_num].update(shot_data)

                        # Mark batch as completed
                        checkpoint.completed_batches.append(batch_id)
                        self.save_checkpoint(checkpoint)

                elif batch_job.status == "failed":
                    logger.error(f"Batch {batch_id} failed")

            except Exception as e:
                logger.error(f"Error checking batch {batch_id} status: {e}")

        # Check if all batches are completed
        if len(checkpoint.completed_batches) == total_batches:
            return all_results
        else:
            logger.info(f"Completed {len(checkpoint.completed_batches)} out of {total_batches} batches")
            logger.info("Run the script again to process remaining batches")
            return None

def processing_listing(listing):
    new_listing = dict()
    for key in listing:
        if key == "listing_id":
            continue
        new_listing[key] = listing[key]
    return new_listing


def compute_shot_accuracy(accuracy):
    shot_wise_accuracy = dict()
    for user_id in accuracy:
        for shot_i in accuracy[user_id]:
            if shot_i not in shot_wise_accuracy:
                shot_wise_accuracy[shot_i] = []
            shot_wise_accuracy[shot_i].append(accuracy[user_id][shot_i][0] / accuracy[user_id][shot_i][1])
    return shot_wise_accuracy


def dump_worst_user_data(final_scores, sorted_user_ids, users, root_dir):
    dump_dir = os.path.join(root_dir, "worst_users")
    os.makedirs(dump_dir, exist_ok=True)
    # copy files from worst_simulation_web
    os.system(f"cp -r worst_simulation_web/* {dump_dir}")

    # Create the data structure
    user_data = {
        "sorted_ids": sorted_user_ids,  # Include the sorting order
        "users": {}  # Store user data in a separate object
    }
    counter = 0
    missed_original_data_index = Counter()

    for user in users:
        user_id = user['id']
        if user_id not in user_data["users"]:
            user_data["users"][user_id] = []

        _history = []
        for response_i, response in enumerate(user["responses"]):
            listing = response["listing"]
            descriptions = response["descriptions"]
            selection = response["selection"]
            user_rating = response["rating"]
            user_reason = response["reason"]

            shot_wise_prediction = {}
            for shot_num in final_scores[user_id]:
                if counter not in final_scores[user_id][shot_num]:
                    assert shot_num > response_i, f"Shot number {shot_num} is less than response index {response_i}"
                    missed_original_data_index[shot_num] += 1
                    continue
                score = final_scores[user_id][shot_num][counter]
                prediction = int(score.score_for_description_1 > score.score_for_description_0)
                shot_wise_prediction[shot_num] = prediction
            average_predictions = sum(shot_wise_prediction.values()) / len(shot_wise_prediction)
            average_label = int(average_predictions > 0.5)

            if average_label != selection:
                _history.append({
                    "listing": listing,
                    "descriptions_0": descriptions[0],
                    "descriptions_1": descriptions[1],
                    "selection": selection,
                    "user_rating": user_rating,
                    "user_reason": user_reason,
                    "shot_wise_prediction": shot_wise_prediction,
                    "original_instance_index": counter,
                    "response_id": response_i,
                    "user_id": user_id
                })
            counter += 1

        user_data["users"][user_id].append(_history)

    # Save the data as JSON
    with open(os.path.join(dump_dir, 'user_data.json'), 'w') as f:
        json.dump(user_data, f, indent=2)

    print(f"Missed original data index: {missed_original_data_index}")


def log_statistics(accuracy, root_dir, final_scores, users, linewidth=5, fontsize=50, error_style='band', error_alpha=0.2, figsize=(20, 15)):
    # set up figure size
    plt.figure(figsize=figsize)
    plt.rc('font', size=fontsize)
    user_accuracy = {k: sum(vv[0] for vv in v.values()) / sum(vv[1] for vv in v.values()) for k, v in accuracy.items()}
    # user_accuracy = {k: sum(vv[0] for vv in v.values()) / sum(vv[1] for vv in v.values()) for k, v in accuracy.items()}
    # plot the histogram
    user_acc_items = list(user_accuracy.items())
    user_acc_items.sort(key=lambda x: x[1])
    worst_user_ids = [x[0] for x in user_acc_items]
    dump_worst_user_data(final_scores, worst_user_ids, users, root_dir)
    plt.hist(user_accuracy.values(), bins=20)
    plt.xlabel("User Simulation Accuracy")
    plt.ylabel("User Count")
    plt.savefig(os.path.join(root_dir, "accuracy_histogram.pdf"))
    plt.clf()
    logger.info(f"User-wise accuracy: {user_accuracy}")
    # count the number of users with non-trivial accuracy
    logger.info(f"Number of users with accuracy larger than 0.5: {1 - len([k for k, v in user_accuracy.items() if v < 0.5]) / len(user_accuracy)}")
    logger.info(f"Number of users with accuracy larger than 0.6: {len([k for k, v in user_accuracy.items() if v > 0.6]) / len(user_accuracy)}")
    logger.info("User-wise accuracy mean: {}".format(np.mean(list(user_accuracy.values()))))
    user_std = np.std(list(user_accuracy.values()))
    logger.info(f"User-wise std: {user_std}")

    shot_wise_accuracy = compute_shot_accuracy(accuracy)
    shot_wise_accuracy_mean = {k: np.mean(v) for k, v in shot_wise_accuracy.items()}
    logger.info(f"Shot-wise accuracy: {shot_wise_accuracy_mean}")
    shot_wise_accuracy_std = {k: np.std(v) for k, v in shot_wise_accuracy.items()}
    logger.info(f"Shot-wise std: {shot_wise_accuracy_std}")
    # plot the shot-wise accuracy as line plot
    xs = list(shot_wise_accuracy_mean.keys())
    xs.sort()
    ys = [shot_wise_accuracy_mean[x] for x in xs]
    yerrs = [shot_wise_accuracy_std[x] for x in xs]
    plt.plot(xs, ys, linewidth=linewidth, marker='o')
    if error_style == 'band':
        plt.fill_between(xs, [y - e for y, e in zip(ys, yerrs)], [y + e for y, e in zip(ys, yerrs)], alpha=error_alpha)
    else:
        plt.errorbar(xs, ys, yerr=yerrs, fmt='none', capsize=5)
    plt.xlabel("#(Shots)")
    plt.ylabel("Shot-wise Accuracy")
    plt.savefig(os.path.join(root_dir, "shotwise_accuracy.pdf"))


def calculate_accuracy(final_scores: Dict, users: List[Dict], predictor: PreferencePredictor) -> Dict:
    """Calculate accuracy from the final scores"""
    accuracy = {}
    answer_sheet = predictor.answer_sheet

    for user in users:
        user_id = user['id']
        if user_id not in final_scores:
            continue

        accuracy[user_id] = {}
        for shot_num, response in enumerate(user["responses"]):
            if shot_num not in final_scores[user_id]:
                continue
            for original_data_index, scores in final_scores[user_id][shot_num].items():
                actual_selection = answer_sheet[original_data_index]
                scores = final_scores[user_id][shot_num][original_data_index]
                predicted_selection = int(scores.score_for_description_1 > scores.score_for_description_0)

                if shot_num not in accuracy[user_id]:
                    accuracy[user_id][shot_num] = [0, 0]  # [correct, total]

                if predicted_selection == actual_selection:
                    accuracy[user_id][shot_num][0] += 1
                accuracy[user_id][shot_num][1] += 1

    return accuracy


async def main(args):
    users = json.load(open(args.data, "r", encoding='utf-8'))

    model_name = args.model_name
    exp_name = args.exp_name.replace("|", ".")
    eval_mode = args.eval_mode
    batch_size = args.batch_size
    force_recompute = args.force_recompute
    stop_shot = args.stop_shot

    if DO_NOT_USE_HISTORY in exp_name:
        logger.info("Do not use history, so we set stop shot to 0")
        args.stop_shot = 0

    suffix = f".{exp_name}.{'llama3' if 'custom' in model_name else model_name}.{eval_mode}"
    root_dir = os.path.join(args.data.split(".")[0] + "_batch_api", suffix[1:])
    os.makedirs(root_dir, exist_ok=True)
    logger.add(os.path.join(root_dir, f"predict_preference.log{suffix}"), rotation="10 MB")

    predictor = PreferencePredictor(
        model_name=args.model_name,
        output_dir=root_dir,
        batch_size=batch_size
    )
    if len(predictor.tasks) == 0 or force_recompute:
        if force_recompute:
            predictor.clear_registry()
            logger.info("Forcing recompute tasks...")
        # Add tasks for each user
        logger.info("Creating batch tasks...")
        counter = 0
        for user in tqdm(users):
            user_id = user['id']
            user_profile = user["profile"]

            # Initialize history for this user
            history = []

            for shot_num, response in enumerate(user["responses"]):
                if shot_num > args.stop_shot:
                    counter += len(user["responses"]) - shot_num
                    break

                # listing = processing_listing(response["listing"])
                listing = response["listing"]
                descriptions = response["descriptions"]

                # Format history info if we're using it
                history_info = ""
                if DO_NOT_USE_HISTORY not in exp_name:
                    if history:
                        history_info = '[Begin of History]\n\n' + '\n\n'.join(history) + '\n\n[End of History]'

                for offline_response_i, offline_response in enumerate(user["responses"][shot_num:]):
                    offline_descriptions = offline_response["descriptions"]
                    offline_listing = offline_response['listing']
                    actual_selection = int(offline_response["selection"])
                    predictor.add_task(
                        user_id=user_id,
                        shot_num=shot_num,
                        test_instance_id=offline_response_i,
                        listing=offline_listing,
                        description_0=offline_descriptions[0],
                        description_1=offline_descriptions[1],
                        user_profile=user_profile,
                        original_data_index=counter + offline_response_i,
                        history=history_info,
                        selection=actual_selection
                    )

                # Update history after adding task
                if DO_NOT_USE_HISTORY not in exp_name:
                    # Since we're doing batch processing, we'll use the actual user selection
                    # for history rather than predicted selection
                    selection = int(response["selection"])
                    user_rating = response["rating"]
                    user_reason = response["reason"]

                    history.append(
                        f"Listing: {processing_listing(listing)}\n\n"
                        f"Description 0: {descriptions[0]}\n\n"
                        f"Description 1: {descriptions[1]}\n\n"
                        f"User Preference: {selection} ({descriptions[selection]})\n\n"
                        f"User Rating: {user_rating}\n\n"
                        f"User Reason: {user_reason}\n\n"
                    )
                counter += 1

    predictor.save_registry()
    logger.info("Total tasks: %d" % len(predictor.tasks))
    # Get or create batch job
    logger.info("Processing batches...")
    final_scores = await predictor.process_batches()

    if final_scores:
        # Calculate accuracy
        accuracy = calculate_accuracy(final_scores, users, predictor)

        # Save results
        torch.save(final_scores, os.path.join(root_dir, f"batch_scores.pt{suffix}"))
        torch.save(accuracy, os.path.join(root_dir, f"batch_accuracy.pt{suffix}"))

        # Log statistics
        log_statistics(accuracy, root_dir, final_scores, users)
    else:
        logger.info("Some batches are still pending. Run the script again to continue processing.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="naive_few_shot")
    parser.add_argument("--eval_mode", type=str, default="online")
    parser.add_argument("--force_recompute", action="store_true")
    parser.add_argument("--stop_shot", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2000)
    args = parser.parse_args()

    asyncio.run(main(args))
