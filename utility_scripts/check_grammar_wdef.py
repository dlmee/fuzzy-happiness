import json
import re

def read_json(filename):
    """Read a JSON file and return the data."""
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def read_dev_txt(text, target = ['Genesis']):
    """Process a text file to extract Devanagari words and calculate frequency."""
    with open(text, encoding="utf-8") as fi:
        lines = fi.readlines()
    # Split lines based on sentence end markers
    lines = [line for line in lines if line.split()[0] in target]
    #lines = re.split("[\.\?\!]", lines)
    # Process each line
    lines = [[re.sub(r"[^\u0900-\u0963\u0965-\u097F]", "", word) for word in re.split(" |-|\n|—", line)] for line in lines]
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

def map_words_to_stems(words, stem_dict, entity, sw, definitions):
    """Map words to their stems, enrich with definitions early, optimize single-entry stems with unknown definitions."""
    final_dict = {}
    for word in words:
        if word in entity or (word in sw and sw[word] > 0.001):
            continue
        if word in stem_dict:
            stem = stem_dict[word][0] if isinstance(stem_dict[word], list) else stem_dict[word]
            if stem not in final_dict:
                final_dict[stem] = {"forms": [word], "definitions": definitions.get(stem, "UNKNOWN")}
            elif word not in final_dict[stem]["forms"]:
                final_dict[stem]["forms"].append(word)

    # Second pass: Adjust single-entry stems only if their definitions are UNKNOWN
    for stem in list(final_dict.keys()):
        if len(final_dict[stem]["forms"]) == 1 and final_dict[stem]["definitions"] == "UNKNOWN":
            word = final_dict[stem]["forms"][0]
            best_match = None
            max_length = 0
            for other_stem in final_dict:
                if len(other_stem) < 3: continue
                if other_stem != stem and other_stem in word and len(other_stem) > max_length:
                    best_match = other_stem
                    max_length = len(other_stem)
            if best_match:
                final_dict[best_match]["forms"].append([word," moved"])
                del final_dict[stem]  # Remove the original stem entry

    # Sort the dictionary by the length of the values' forms in descending order
    sorted_final_dict = {k: v for k, v in sorted(final_dict.items(), key=lambda item: len(item[1]["forms"]), reverse=True)}
    
    # Insert the length at the top of the dictionary
    sorted_final_dict = {'**length**': len(sorted_final_dict)} | sorted_final_dict

    return sorted_final_dict





# Example usage
input_text = 'data/source_texts/nepali_bible_reformatted.txt'  # Path to the text file to be processed
target_books = ["Acts", "Romans", "1 Corinthians", "2 Corinthians", "Galatians","Ephesians","Philippians","Colossians",
"1 Thessalonians", "2 Thessalonians","1 Timothy","2 Timothy",
"Titus","Philemon","Hebrews","James","1 Peter","2 Peter",
"1 John","2 John","3 John","Jude","Revelation", "Genesis", "Matthew", "Mark", "Luke", "John"]
entity_file = 'data/extracted_entities.json'
stopword_file = 'data/stopword_analysis.json'
input_file = 'new_best_grammar.json'  # Path to your JSON file
output_file = 'new_dictionary_afx_tdef.json'  # Path to output JSON containing stems and words mapping
definitions = 'data/nep3dict_unified_wdef.json'

# Read and process the JSON file to create the stem dictionary
data = read_json(input_file)
entity = read_json(entity_file)
definitions = read_json(definitions)
sw = read_json(stopword_file)
stem_dict = {}
for entry in data['combinations']:
    if entry['generates'] and entry['stems']:
        if len(entry['generates']) > 1 and len(entry['generates']) < 10:
            #print("found you!")
            pass
        for word in entry['generates']:
            key = re.sub('-', '', word)
            value = entry['stems']
            stem_dict[key] = value


# Process the text file and get words
words, _ = read_dev_txt(input_text, target_books)

# Map words to their corresponding stems
final_dictionary = map_words_to_stems(words, stem_dict, entity, sw, definitions)

# Optionally, write the final dictionary to a JSON file
with open(output_file, 'w', encoding='utf-8') as file:
    json.dump(final_dictionary, file, indent=4, ensure_ascii=False)

print("Mapping complete. Check the output file for results.")
