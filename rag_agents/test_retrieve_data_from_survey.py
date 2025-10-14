import csv
import json
from utils import get_retrieval_data

if __name__ == '__main__':
    all_data = get_retrieval_data("./data/ai_realtor_listing_data.json")
    url_buffer = dict()
    for data in all_data:
        # assert data['url'] not in url_buffer
        # if data['url'] in url_buffer:
        #     assert data == url_buffer[data['url']], f"Duplicate URL: {data['url']}"
        try:
            key_url = (data["url"], data['street_address'])
            assert key_url not in url_buffer, f"Duplicate URL: {key_url}, data_id: {data['id']}, observed: {url_buffer[key_url]['id']}"
            url_buffer[key_url] = data
        except Exception as e:
            print(e)
            # print(data.keys())
            # print stacktrace
            import sys
            import traceback
            traceback.print_exc(file=sys.stdout)
            # exit()

    for resample_id in [0, 1]:
        resample_data = []
        matched_count = 0
        url_match_count = 0
        mismatched_items = []
        complete_original_data = []
        with open(f"../data/resample_batch_{resample_id}.csv", "r") as f_in:
            reader = csv.DictReader(f_in)
            for row in reader:
                resample_data.append(row)
                key_url = (row["url"], row['address'])
                if key_url in url_buffer:
                    url_match_count += 1
                    flag = True
                    for key in row:
                        if key == "ID":
                            continue
                        else:
                            if key == "neighborhood":
                                if str(row[key]) != str(url_buffer[key_url]['neighborhood_region']):
                                    flag = False
                                    mismatched_items.append((row['url'], key, row[key], url_buffer[key_url]['neighborhood_region']))
                                    break
                            elif key == "address":
                                if str(row[key]) != str(url_buffer[key_url]['street_address']):
                                    flag = False
                                    mismatched_items.append((row['url'], key, row[key], url_buffer[key_url]['street_address']))
                                    break
                            elif str(row[key]) != str(url_buffer[key_url][key]):
                                flag = False
                                mismatched_items.append((row['url'], key, row[key], url_buffer[key_url][key]))
                                break
                    if flag:
                        matched_count += 1
                        complete_original_data.append(url_buffer[key_url])
        print(f"Resample {resample_id}, Matched {matched_count}/{len(resample_data)}")
        print(f"URL Matched {url_match_count}/{len(resample_data)}")
        print(f"Mismatched Items: {mismatched_items}")
        with open(f"../data/resample_batch_{resample_id}_input.json", "w") as f_out:
            json.dump(resample_data, f_out, indent=4)



