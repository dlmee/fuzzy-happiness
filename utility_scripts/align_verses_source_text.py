import json
import re
from tqdm import tqdm

def read_verses(file_path):
    verses = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) < 2: continue
            verse_ref = parts[0]
            verse_text = parts[1]
            verses[verse_ref] = verse_text
    return verses

def align_verses(english_file, nepali_file):
    english_verses = read_verses(english_file)
    nepali_verses = read_verses(nepali_file)

    aligned_verses = {}

    for nepali_verse, nepali_text in tqdm(nepali_verses.items()):
        nepali_verse_split = re.split(" |:", nepali_verse)
        if "-" not in nepali_verse_split[2]: continue
        nepali_book, nepali_chapter, nepali_verse_num = nepali_verse_split[0], nepali_verse_split[1], nepali_verse_split[2]

        if '-' in nepali_verse_num:
            start_verse, end_verse = map(int, nepali_verse_num.split('-'))
        else:
            start_verse = end_verse = int(nepali_verse_num)

        aligned_verse = {}
        for english_verse, english_text in english_verses.items():
            english_verse_split = re.split(" |:", english_verse)
            english_book, english_chapter, english_verse_num = english_verse_split[0], english_verse_split[1], english_verse_split[2]
            if (english_book == nepali_book and english_chapter == nepali_chapter and 
                start_verse <= int(english_verse_num) <= end_verse):
                aligned_verse['eng ' + english_verse_num] = english_text
                aligned_verse['nep ' + nepali_verse_num] = nepali_text
        aligned_verses[nepali_verse] = aligned_verse

    return aligned_verses

# Example usage
english_file = 'data/source_texts/niv.txt'
nepali_file = 'data/source_texts/nepali_bible_reformatted.txt'
aligned_verses = align_verses(english_file, nepali_file)


# Output the aligned verses as JSON
output_file = 'aligned_verses.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(aligned_verses, f, ensure_ascii=False, indent=4)
