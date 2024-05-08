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

def analyze_affixes(data):
    """Analyze the suffixes and prefixes in the data and count their occurrences, returning sorted results."""
    suffix_counts = {}
    prefix_counts = {}

    # Access the list of elements in 'combinations'
    for entry in data['combinations']:
        # Process suffixes
        suffixes = entry.get('suffixes', [])
        if suffixes:
            for suffix in suffixes:
                if suffix not in suffix_counts:
                    suffix_counts[suffix] = 1
                else:
                    suffix_counts[suffix] += 1

        # Process prefixes
        prefixes = entry.get('prefixes', [])
        if prefixes:
            for prefix in prefixes:
                if prefix not in prefix_counts:
                    prefix_counts[prefix] = 1
                else:
                    prefix_counts[prefix] += 1

    # Sort the suffixes and prefixes by count in descending order
    sorted_suffix_counts = dict(sorted(suffix_counts.items(), key=lambda item: item[1], reverse=True))
    sorted_prefix_counts = dict(sorted(prefix_counts.items(), key=lambda item: item[1], reverse=True))

    return sorted_suffix_counts, sorted_prefix_counts


def process_affixes(file_path, output_file):
    """Process the file to analyze affixes and write the results to a JSON file."""
    # Read data from the JSON file
    data = read_json(file_path)

    # Analyze suffixes and prefixes
    suffix_counts, prefix_counts = analyze_affixes(data)

    # Combine results into one dictionary to write to the file
    results = {
        'suffixes': suffix_counts,
        'prefixes': prefix_counts
    }

    # Write results to a new JSON file
    write_json(results, output_file)

# Usage
file_path = 'new_best_grammar.json'  # Path to the input JSON file
output_file = 'data/affix_counts.json'  # Path to the output JSON file
process_affixes(file_path, output_file)
