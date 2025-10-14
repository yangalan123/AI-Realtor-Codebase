from utils import load_hf_dataset
if __name__ == '__main__':
    original_hf_ds = load_hf_dataset(filter_by_city="chicago")
    print("Number of records in the dataset: ", len(original_hf_ds))
    # save to json
    original_hf_ds.to_json("chicago_dataset.json")