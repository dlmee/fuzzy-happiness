import json
import re

def read_json(filename):
    """Read a JSON file and return the data."""
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def read_dev_txt(text):
    """Process a text file to extract Devanagari words and calculate frequency."""
    with open(text, encoding="utf-8") as fi:
        lines = fi.read()
    # Split lines based on sentence end markers
    lines = re.split("[\.\?\!]", lines)
    # Process each line
    lines = [[re.sub(r"[^\u0900-\u097F]", "", word) for word in re.split(" |-|\n|—", line)] for line in lines]
    # Calculate word frequencies
    counts = {'**total**': 0}
    for line in lines:
        for word in line:
            if not word: continue
            if word not in counts:
                counts[word] = 1
            else:
                counts[word] += 1
            counts['**total**'] += 1
    # Normalize frequencies
    for k in list(counts.keys()):
        if k != '**total**':
            counts[k] = counts[k] / counts['**total**']
    counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    counts = {word[0]: word[1] for word in counts}
    # Extract words set
    lines = set([word for line in lines for word in line if word])
    return list(lines), counts

def map_words_to_stems(words, stem_dict):
    """Map words to their stems using the provided dictionary."""
    final_dict = {'**length**':0}
    for word in words:
        if word in stem_dict:
            if type(stem_dict[word]) == list:
                stem_dict[word] = stem_dict[word][0]
            if stem_dict[word] not in final_dict:
                final_dict[stem_dict[word]] = [word] # Remove hyphens and look up the stem
            else:
                if word in final_dict[stem_dict[word]]:
                    continue
                else:
                    final_dict[stem_dict[word]].append(word)
    final_dict['**length**'] = len(final_dict.keys()) - 1
    return final_dict

# Example usage
input_text = 'data/source_texts/nepali_bible_reformatted.txt'  # Path to the text file to be processed
input_file = 'new_best_grammar.json'  # Path to your JSON file
output_file = 'new_dictionary.json'  # Path to output JSON containing stems and words mapping

# Read and process the JSON file to create the stem dictionary
data = read_json(input_file)
stem_dict = {}
for entry in data['combinations']:
    if entry['generates'] and entry['stems']:
        if len(entry['generates']) > 1 and len(entry['generates']) < 10:
            print("found you!")
        for word in entry['generates']:
            key = re.sub('-', '', word)
            value = entry['stems']
            stem_dict[key] = value


# Process the text file and get words
words, _ = read_dev_txt(input_text)

# Map words to their corresponding stems
final_dictionary = map_words_to_stems(words, stem_dict)

# Optionally, write the final dictionary to a JSON file
with open(output_file, 'w', encoding='utf-8') as file:
    json.dump(final_dictionary, file, indent=4, ensure_ascii=False)

print("Mapping complete. Check the output file for results.")
