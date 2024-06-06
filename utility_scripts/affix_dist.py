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
                if suffix == 'Null' or suffix == 0: continue
                if suffix not in suffix_counts:
                    suffix_counts[suffix] = 1
                else:
                    suffix_counts[suffix] += 1

        # Process prefixes
        prefixes = entry.get('prefixes', [])
        if prefixes:
            for prefix in prefixes:
                if prefix == 'Null' or prefix == 0: continue
                if prefix not in prefix_counts:
                    prefix_counts[prefix] = 1
                else:
                    prefix_counts[prefix] += 1

    # Sort the suffixes and prefixes by count in descending order
    sorted_suffix_counts = dict(sorted(suffix_counts.items(), key=lambda item: item[1], reverse=True))
    sorted_prefix_counts = dict(sorted(prefix_counts.items(), key=lambda item: item[1], reverse=True))

    return sorted_suffix_counts, sorted_prefix_counts

def create_parse(data, suffix, prefix, llmdata):
    stems = {stem['stems']:stem for stem in data['combinations'] if type(stem['stems']) == str}
    parsed = {}
    for k, v in llmdata.items():
        for word in v['forms']:
            if k in stems:
                entry = stems[k]
                stem = entry.get('stems', "UNKNOWN")
                if type(stem) == list:
                    stem = stem[0]
                generated = entry.get('generates', [])
                if len(generated) > 1:

                    parsed[stem] = {re.sub("-", "", g):{'prefix':list({fix for fix in g.split(stem)[0].split('-') if fix in prefix and prefix[fix] > 3 and fix != stem}), 'suffix':list({fix for fix in g.split(stem)[1].split('-') if fix in suffix and suffix[fix] > 3 and fix != stem})} for g in generated if '-' in g}
            else:
                affixes = [elem for elem in word.split(k) if elem]
                if len(affixes) > 1:
                    prefixes = [pre for pre in affixes if not word.split(pre)[0]]
                    suffixes = [suf for suf in affixes if not word.split(suf)[-1]]  
                    if k not in parsed:
                        parsed[k] = {word:{'prefixes':prefixes, 'suffixes':suffixes}}
                    else:
                        if word not in parsed[k]:
                            parsed[k][word] = {}
                        parsed[k][word]['prefixes'] = prefixes
                        parsed[k][word]['suffixes'] = suffixes

    return parsed




def process_affixes(file_path, output_file, llm_file):
    """Process the file to analyze affixes and write the results to a JSON file."""
    # Read data from the JSON file
    data = read_json(file_path)
    data2 = read_json(llm_file)

    # Analyze suffixes and prefixes
    suffix_counts, prefix_counts = analyze_affixes(data)

    parse = create_parse(data, suffix_counts, prefix_counts, data2)
    # Combine results into one dictionary to write to the file
    results = {
        'suffixes': suffix_counts,
        'prefixes': prefix_counts
    }

    # Write results to a new JSON file
    write_json(parse, output_file)

# Usage
file_path = 'new_best_grammar.json'  # Path to the input JSON file
llm_path = 'swahili/formatted_llm_response_ALL.json'
output_file = 'swahili/swh_parse.json'  # Path to the output JSON file
process_affixes(file_path, output_file, llm_path)
