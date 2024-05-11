import json
import re

def read_json(filename):
    """Read a JSON file and return the data."""
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def write_json(data, filename):
    """Write data to a JSON file."""
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def merge_dictionaries(file1, file2, output_file):
    """Merge two JSON files into one dictionary."""
    # Read the dictionaries from the files
    dict1 = read_json(file1)
    dict2 = read_json(file2)

    # Unify the dictionaries
    #unified_dict = dict1.copy()  # Start with a copy of the first dictionary
    unified_dict = {re.sub("[^a-z]", "", v['Word'].strip().lower()):[v['Meaning']] for v in dict1.values()}
    """for key, value in dict2.items():
        if key in unified_dict:
            # Append the second dictionary's values to the existing values list
            if isinstance(unified_dict[key], list):
                unified_dict[key].extend(value if isinstance(value, list) else [value])
            else:
                unified_dict[key] = [unified_dict[key], value] if not isinstance(value, list) else [unified_dict[key]] + value
        else:
            # Add the missing key and value
            unified_dict[key] = value"""
    pattern = r'\b(?:[a-zA-Z]{2,})\b'

    for elem in dict2:
        if elem['w'] in unified_dict:
            unified_dict[elem['w']].append(' '.join(re.findall(pattern, re.sub(r'<[^>]*>', '', elem['h']))))
        else:
            unified_dict[elem['w']] = [' '.join(re.findall(pattern, re.sub(r'<[^>]*>', '', elem['h'])))]
    # Write the unified dictionary to the output file
    unified_dict = list(unified_dict.keys())
    write_json(unified_dict, output_file)

# Usage
file1 = 'data/source_texts/sw_dict.json'
file2 = 'data/source_texts/sw-eng.js'
output_file = 'data/sw_dict_unified_wdef.json'
merge_dictionaries(file1, file2, output_file)
