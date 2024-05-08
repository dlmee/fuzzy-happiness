import json

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
    unified_dict = dict1.copy()  # Start with a copy of the first dictionary
    for key, value in dict2.items():
        if key in unified_dict:
            # Append the second dictionary's values to the existing values list
            if isinstance(unified_dict[key], list):
                unified_dict[key].extend(value if isinstance(value, list) else [value])
            else:
                unified_dict[key] = [unified_dict[key], value] if not isinstance(value, list) else [unified_dict[key]] + value
        else:
            # Add the missing key and value
            unified_dict[key] = value

    # Write the unified dictionary to the output file
    write_json(unified_dict, output_file)

# Usage
file1 = 'data/nep_supp_wdef.json'
file2 = 'data/nep_wdef.json'
output_file = 'data/nep_dict_unified_wdef.json'
merge_dictionaries(file1, file2, output_file)
