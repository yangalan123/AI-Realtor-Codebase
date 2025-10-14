import json
import traceback
import csv
import glob

if __name__ == '__main__':
    # filenames = glob.glob("./rag_output_resampled*.json")
    filenames = glob.glob("./rag_output/resampled*.json")
    for filename in filenames:
        all_data = []
        try:
            with open(filename, "r") as f_in:
                for line in f_in:
                    all_data.append(json.loads(line))
            csv_filename = filename.replace(".json", ".csv")
            # only keep "address", "price" and "description"
            with open(csv_filename, "w") as f_out:
                writer = csv.DictWriter(f_out, fieldnames=["address", "price", "description"])
                writer.writeheader()
                for data in all_data:
                    descriptions = data["description"].split("\n\n")[0].split("\n")
                    descriptions = [d.strip() for d in descriptions if len(d.strip()) > 0]
                    # remove descriptions if it only contains punctuation
                    descriptions = [d for d in descriptions if not all([c in ".,:;?!-" for c in d])]
                    description = " ".join(descriptions)
                    writer.writerow({"address": data["address"] if "address" in data else data['street_address'], "price": data["price"], "description": description})
        except:
            print(f"error happened when processing, skipped: {filename}")
            # print stacktrace
            traceback.print_exc()

