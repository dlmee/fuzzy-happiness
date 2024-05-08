import json

def process_data(file_path):
    # Read the text file
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.readlines()

    # Convert string representation of list of tuples into actual list of tuples
    data = [eval(line) for line in data]

    # Dictionary for storing stem as key and other forms as subkeys
    stem_dict = {}
    # Set for storing all unique words
    all_words = set()

    for line in data:
        for original, stem, lemma in line:
            if original not in stem_dict:
                stem_dict[original] = {'stem': stem, 'lemma': lemma}

            # Add words to the set of all words
            all_words.add(original)

    # Save the dictionary in JSON format
    with open('stem_test.json', 'w', encoding='utf-8') as json_file:
        json.dump(stem_dict, json_file, ensure_ascii=False, indent=4)

    # Save all words as a list in JSON format
    with open('test_words.json', 'w', encoding='utf-8') as json_file:
        json.dump(list(all_words), json_file, ensure_ascii=False, indent=4)

# Example usage
process_data('data/nepali_stem_test.txt')
