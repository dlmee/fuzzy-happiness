import json

def read_json(filename):
    """Read a JSON file and return the data."""
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def write_json(data, filename):
    """Write data to a JSON file."""
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def merge_and_sort_json_files(file1, file2, output_file):
    """Merge two JSON files, remove duplicates, and write sorted output."""
    # Read the data from the files
    data1 = read_json(file1)
    data2 = read_json(file2)

    # Merge the data into a set to remove duplicates
    unified_set = set(data1 + data2)

    # Sort the data by the length of the elements, longest first
    sorted_list = sorted(unified_set, key=len, reverse=True)

    # Write the sorted data to the output file
    write_json(sorted_list, output_file)

# Usage
file1 = 'data/nep_dict_1.json'
file2 = 'data/nep_dict_supp.json'
output_file = 'data/nep_dict_unified.json'
merge_and_sort_json_files(file1, file2, output_file)
