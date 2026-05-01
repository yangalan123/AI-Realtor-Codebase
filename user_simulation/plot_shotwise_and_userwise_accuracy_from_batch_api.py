# in case old batches are not available at OpenAI, separate the plotting codes out for the paper visualization
import torch
import json
import os
from user_simulation.predicting_preference_batch_api import log_statistics

data = "responses_latest.json"
users = json.load(open(data, "r", encoding='utf-8'))
exp_name = "naive_few_shot"
model_name = "gpt-4o-mini"
eval_mode = "offline"
suffix = f".{exp_name}.{'llama3' if 'custom' in model_name else model_name}.{eval_mode}"
root_dir = os.path.join(data.split(".")[0] + "_batch_api", suffix[1:])
final_scores = torch.load(os.path.join(root_dir, f"batch_scores.pt{suffix}"))
accuracy = torch.load(os.path.join(root_dir, f"batch_accuracy.pt{suffix}"))

# Log statistics
figsize = (28, 23)
fontsize = 65
# plt.rcParams.update({
#     'font.family': 'sans-serif',
#     'font.weight': 'bold',  # Changed to normal weight to match reference
#     'axes.labelweight': 'bold',
#     'axes.titleweight': 'bold'
# })
log_statistics(accuracy, root_dir, final_scores, users, figsize=figsize, fontsize=fontsize)
