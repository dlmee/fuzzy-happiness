import json
from collections import defaultdict

# Load data from JSON file
with open('data/sw_dict_unified_wdef.json', 'r') as file:
    data = json.load(file)

result = defaultdict(list)

for key1 in data.keys():
    # Iterate over each other key
    for key2 in data.keys():
        if key1 != key2:  # Ensure we're not comparing a key against itself
            # For each sentence in key2, find occurrences of key1
            if key1 in key2:
                result[key1].append(key2)


# Sort the result dictionary by the length of the values (keys that fit into each key)
sorted_result = dict(sorted(result.items(), key=lambda x: len(x[1]), reverse=True))

with open("data/simple_sw_analysis.json", "w") as outj:
    json.dump(result, outj, indent=4, ensure_ascii=False)
