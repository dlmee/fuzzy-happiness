import json
import re
import unicodedata

def read_json(filename):
    """Read a JSON file and return the data."""
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def clean_text(text):
    # Regex to remove invisible characters
    #return re.sub(r'\s+', ' ', text).strip()
    return unicodedata.normalize('NFC', text)

def read_dev_txt(text, target = ['Genesis']):
    """Process a text file to extract Devanagari words and calculate frequency."""
    with open(text, encoding="utf-8") as fi:
        lines = fi.readlines()
    # Split lines based on sentence end markers
    lines = [line for line in lines if line.split()[0] in target]
    #lines = re.split("[\.\?\!]", lines)
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

def map_words_to_stems(words, stem_dict, entity = [], sw = [], second=None):
    """Map words to their stems, optimize single-entry stems, sort by length of values, and track changes."""
    final_dict = {"**score**":["**score**"]}
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
    if second:
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
                    final_dict[best_match].append(word)
                    del final_dict[stem]  # Remove the original stem entry

    # Sort the dictionary by the length of the values in descending order
    #sorted_final_dict = {k: v for k, v in sorted(final_dict.items(), key=lambda item: len(item[1]), reverse=True)}
    test_final_dict = {word:k for k,v in final_dict.items() for word in v}
    # Insert the length at the top of the dictionary
    #sorted_final_dict = {'**length**': len(sorted_final_dict)} | sorted_final_dict

    return test_final_dict



# Example usage
test_data = 'test_words.json'
validator = 'stem_test.json'
input_file = 'new_best_grammar.json'  # Path to your JSON file
output_file = 'test_dictionary_V5.json'  # Path to output JSON containing stems and words mapping

# Read and process the JSON file to create the stem dictionary
data = read_json(input_file)
test_data = read_json(test_data)
validator = read_json(validator)
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

words = test_data

# Map words to their corresponding stems
score = []
final_dictionary = map_words_to_stems(words, stem_dict)
for k,v in validator.items():
    if k not in final_dictionary:
        print("MISSING")
        continue
    if clean_text(v['stem']) == clean_text(final_dictionary[k]) or clean_text(v['lemma']) == clean_text(final_dictionary[k]):
        print("YES!")
        score.append(1)
    else:
        print(f"It should have been {v['stem']} but instead it was {final_dictionary[k]}")
        score.append(0)

final_dictionary['**score**'] = sum(score)/len(score)
print(f"The final score is {sum(score)/len(score)}")



# Optionally, write the final dictionary to a JSON file
with open(output_file, 'w', encoding='utf-8') as file:
    json.dump(final_dictionary, file, indent=4, ensure_ascii=False)

print("Mapping complete. Check the output file for results.")
