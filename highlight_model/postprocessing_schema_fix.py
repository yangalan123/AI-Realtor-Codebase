from const import DESIRED_FEATURE_NAMES, POST_PROCESSING_BANNED_LOW_FREQUENCY_FEATURES
import evaluate
import torch
if __name__ == '__main__':
    filename = "./checkpoints/feature_classifier_best_predictions.pt"
    predictions, references = torch.load(filename)
    assert len(predictions) == len(references), "The number of predictions and references are not the same."
    assert len(predictions) % len(DESIRED_FEATURE_NAMES) == 0, "The number of predictions is not a multiple of the number of desired features."
    predictions = predictions.reshape(-1, len(DESIRED_FEATURE_NAMES))
    references = references.reshape(-1, len(DESIRED_FEATURE_NAMES))
    processed_pred = []
    processed_ref = []
    acc_metric = evaluate.combine(['accuracy', 'f1', 'precision', 'recall'])
    # we reuse accuracy metric for em, as our output is not string-string exact match so we cannot use original EM
    # em_metric = evaluate.load("accuracy")
    for data_i in range(len(predictions)):
        for feature_i in range(len(DESIRED_FEATURE_NAMES)):
            feature_name = DESIRED_FEATURE_NAMES[feature_i]
            if feature_name in POST_PROCESSING_BANNED_LOW_FREQUENCY_FEATURES:
                continue
            processed_pred.append(predictions[data_i][feature_i])
            processed_ref.append(references[data_i][feature_i])
    acc_metric.add_batch(predictions=processed_pred, references=processed_ref)
    print(acc_metric.compute())