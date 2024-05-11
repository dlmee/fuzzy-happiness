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

def read_txt(self, corpus):
    text = corpus
    counts = {'**total**': 0}
    with open(text, encoding="utf-8") as fi:
        lines = fi.read()

    lines = re.split(r"[\.\?\!]", lines)
    processed_lines = []

    for line in lines:
        line = line.split('\t')
        if len(line) <= 1: continue
        words = [re.sub("[^a-z]", "", word.lower()) for word in re.split(" |-|\n|—", line[1])]
        processed_lines = processed_lines + [word for word in words if word]
        for word in words:
            if word in counts:
                counts[word] += 1
            else:
                counts[word] = 1
            counts['**total**'] += 1
    
    return list(set(processed_lines)), counts


def map_words_to_stems(words, stem_dict, entity, sw):
    """Map words to their stems, optimize single-entry stems, sort by length of values, and track changes."""
    final_dict = {}
    for word in words:
        if word in entity or (word in sw and sw[word] > 0.001):
            continue
        if word in stem_dict:
            stem = stem_dict[word][0] if isinstance(stem_dict[word], list) else stem_dict[word]
            if stem not in final_dict:
                final_dict[stem] = [word]
            elif word not in final_dict[stem]:
                final_dict[stem].append(word)

    # Second pass: Adjust single-entry stems
    for stem in list(final_dict.keys()):
        if len(final_dict[stem]) == 1:
            word = final_dict[stem][0]
            best_match = None
            max_length = 0
            # Find the longest stem key that is a substring of word
            for other_stem in final_dict:
                if len(other_stem) < 3: continue
                if other_stem != stem and other_stem in word and len(other_stem) > max_length:
                    best_match = other_stem
                    max_length = len(other_stem)
            if best_match:
                final_dict[best_match].append(word + " moved")
                del final_dict[stem]  # Remove the original stem entry

    # Sort the dictionary by the length of the values in descending order
    sorted_final_dict = {k: v for k, v in sorted(final_dict.items(), key=lambda item: len(item[1]), reverse=True)}
    
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
output_file = 'new_dictionary.json'  # Path to output JSON containing stems and words mapping
input_file = 'data/side_hustle.json'  # Path to your JSON file
output_file = 'simple_dictionary.json'  # Path to output JSON containing stems and words mapping

# Read and process the JSON file to create the stem dictionary
data = read_json(input_file)
stem_dict = {}
for k,v in data.items():
    if len(v) > 40:
        for word in v:
            stem_dict[word] = word
    else:
        for word in v:
            stem_dict[word] = k
print(len(stem_dict))
            
entity = read_json(entity_file)
sw = read_json(stopword_file)
# Process the text file and get words
words, _ = read_dev_txt(input_text)

# Map words to their corresponding stems
final_dictionary = map_words_to_stems(words, stem_dict, entity, sw)

# Optionally, write the final dictionary to a JSON file
with open(output_file, 'w', encoding='utf-8') as file:
    json.dump(final_dictionary, file, indent=4, ensure_ascii=False)

print("Mapping complete. Check the output file for results.")
