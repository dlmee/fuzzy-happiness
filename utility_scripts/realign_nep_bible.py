import json

def read_verses_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def read_verses_from_txt(file_path):
    verses = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) <= 1: continue
            verse_ref_parts = parts[0].split()
            verse_ref = " ".join(verse_ref_parts[:-1]) + " " + verse_ref_parts[-1].split(":")[0]
            verse_num = verse_ref_parts[-1].split(":")[1].split()[0]  # Extracting verse number
            verse_text = parts[1]
            verses[f'{verse_ref}:{verse_num}'] = verse_text
    return verses



def break_out_verses(verses_json, verses_txt):
    broken_out_verses = {}

    for verse_ref, verse_text in verses_txt.items():
        if verse_ref in verses_json:
            for sub_key, sub_value in verses_json[verse_ref].items():
                verse_num = sub_key.split()[1]
                book = verse_ref.split()
                chap = book[1].split(':')
                broken_key = f"{book[0]} {chap[0]}:{verse_num}"
                broken_out_verses[broken_key] = sub_value
        else:
            # Prepare the verse to be written out in text format
            broken_out_verses[verse_ref] = verse_text

    return broken_out_verses



def write_broken_out_verses_to_txt(broken_out_verses, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for key, value in broken_out_verses.items():
            file.write(key + '\t' + value + '\n')

# Paths to input files
json_file_path = 'My_LLM_Results_aligned.json'
txt_file_path = 'data/source_texts/nepali_bible_reformatted.txt'

# Read verses from input files
verses_json = read_verses_from_json(json_file_path)
verses_txt = read_verses_from_txt(txt_file_path)

# Break out verses and write to output file
broken_out_verses = break_out_verses(verses_json, verses_txt)
output_txt_file_path = 'broken_out_verses.txt'
write_broken_out_verses_to_txt(broken_out_verses, output_txt_file_path)
