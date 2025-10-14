import json
from utils import get_original_all_features_data

all_features = get_original_all_features_data()
# convert a dict to a list
buf = list(all_features.values())

state = "WA"
city = "Seattle"
home_type = "SINGLE_FAMILY"
bedrooms = 4.0
bathrooms = 3.0

# filter data according to the conditions, iteratively, print out the number of data left at each iteration
filter1 = [item for item in buf if item["state"] == state]
print(f"filter1: {len(filter1)}")
filter2 = [item for item in filter1 if item["city"] == city]
print(f"filter2: {len(filter2)}")
filter3 = [item for item in filter2 if item["home_type"] == home_type]
print(f"filter3: {len(filter3)}")
filter4 = [item for item in filter3 if item["bedrooms"] >= bedrooms]
print(f"filter4: {len(filter4)}")
filter5 = [item for item in filter4 if item["bathrooms"] >= bathrooms]
print(f"filter5: {len(filter5)}")

