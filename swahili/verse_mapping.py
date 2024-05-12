import re
import json

def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return lines

def load_mapping_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    return mapping

import re

def extract_verse_stems(text_lines, target_books, word_mapping):
    verse_stems = {}
    stem_verses = {}

    current_book = None
    for line in text_lines:
        match = re.match(r'^(\d*\s*\S+)\s+(\d+:\d+(?:-\d+)?)\s+(.*)$', line)
        if match:
            book, verse, content = match.groups()
            current_book = book

            if current_book in target_books:
                verse_key = f"{book} {verse}"
                verse_stems[verse_key] = []

                words = content.split()
                for word in words:
                    # Remove any non-alphanumeric characters from the word
                    word = re.sub(r'[^a-zA-Z0-9\u0900-\u097F]+', '', word)
                    if word in word_mapping:
                        stem = word_mapping[word]['stem']
                        if stem not in verse_stems[verse_key]:
                            verse_stems[verse_key].append(stem)

                        if stem not in stem_verses:
                            stem_verses[stem] = []
                        if verse_key not in stem_verses[stem]:
                            stem_verses[stem].append(verse_key)
        else:
            match = re.match(r'^(\S+)\s+(\d+:\d+(?:-\d+)?)\s+(.*)$', line)
            if match:
                book, verse, content = match.groups()
                current_book = book

                if current_book in target_books:
                    verse_key = f"{book} {verse}"
                    verse_stems[verse_key] = []

                    words = content.split()
                    for word in words:
                        # Remove any non-alphanumeric characters from the word
                        word = re.sub(r'[^a-zA-Z0-9\u0900-\u097F]+', '', word)
                        if word in word_mapping:
                            stem = word_mapping[word]['stem']
                            if stem not in verse_stems[verse_key]:
                                verse_stems[verse_key].append(stem)

                            if stem not in stem_verses:
                                stem_verses[stem] = []
                            if verse_key not in stem_verses[stem]:
                                stem_verses[stem].append(verse_key)
    
    return verse_stems, stem_verses


def write_to_json(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    text_file_path = "swahili/swh_mft_reformatted.txt"  # Replace with the path to your text file
    mapping_file_path = "swahili/inverse_final_swahili.json"  # Replace with the path to your mapping file
    target_books = ["Acts", "Romans", "1 Corinthians", "2 Corinthians", "Galatians","Ephesians","Philippians","Colossians",
"1 Thessalonians", "2 Thessalonians","1 Timothy","2 Timothy",
"Titus","Philemon","Hebrews","James","1 Peter","2 Peter",
"1 John","2 John","3 John","Jude","Revelation", "Genesis", "Matthew", "Mark", "Luke", "John"] # Specify the target books

    text_lines = load_text_file(text_file_path)
    word_mapping = load_mapping_file(mapping_file_path)
    verse_stems, stem_verses = extract_verse_stems(text_lines, target_books, word_mapping)
    
    output_file_verse_stems = "verse_stems.json"  # Replace with the desired output file name for verse stems
    output_file_stem_verses = "stem_verses.json"  # Replace with the desired output file name for stem verses

    write_to_json(verse_stems, output_file_verse_stems)
    write_to_json(stem_verses, output_file_stem_verses)
