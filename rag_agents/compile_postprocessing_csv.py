import csv
import copy
import glob

if __name__ == '__main__':
    model_name = "Mixtral-8x7B-Instruct-v0.1"
    for batch_id in [0, 1]:
        csv_filename_pattern = f"./rag_output/*batch_{batch_id}*{model_name}*.csv"
        mode = "rag"
        rule_out_fn_parts = set()
        if "no_retrieval" in mode:
            rule_out_fn_parts.add("no_retrieval")
        if "gpt4" in mode:
            rule_out_fn_parts.add("gpt4")
        all_data_dict = dict()
        suffix_list = []
        for filename in glob.glob(csv_filename_pattern):
            # first check if the file contains the parts we want to rule out
            if any([part in filename for part in rule_out_fn_parts]):
                continue
            with open(filename, "r") as f_in:
                reader = csv.DictReader(f_in)
                suffix = filename.split(model_name)[1].split(".csv")[0]
                assert suffix not in suffix_list, f"suffix {suffix} already exists"
                suffix_list.append(suffix)
                for row_i, row in enumerate(reader):
                    if row_i not in all_data_dict:
                        all_data_dict[row_i] = copy.deepcopy(row)
                        del all_data_dict[row_i]["description"]
                    all_data_dict[row_i][f"{suffix}"] = row["description"]
                    for key in row:
                        if key == "description":
                            continue
                        else:
                            assert all_data_dict[row_i][key] == row[key], f"key {key} not equal"
        with open(f"./rag_output/combined_batch_{batch_id}_{model_name}_{mode}.csv", "w", newline='') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=["address", "price"] + suffix_list)
            writer.writeheader()
            for row_i in all_data_dict:
                writer.writerow(all_data_dict[row_i])
