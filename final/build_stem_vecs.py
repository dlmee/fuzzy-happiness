import json
from collections import defaultdict

def load_stem_verse_mappings(file_paths):
    all_mappings = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
            all_mappings.append(mapping)
    return all_mappings

def build_verse_index(stem_verse_mappings):
    verse_index = {}
    verse_counter = 0
    for mapping in stem_verse_mappings:
        for stem, verses in mapping.items():
            for verse in verses:
                if verse not in verse_index:
                    verse_index[verse] = verse_counter
                    verse_counter += 1
    return verse_index

def build_vectors(stem_verse_mappings, verse_index):
    vectors = {}
    for mapping in stem_verse_mappings:
        for stem, verses in mapping.items():
            vectors[stem] = [0] * len(verse_index)
            for verse in verses:
                vectors[stem][verse_index[verse]] = 1
    return vectors

def write_to_json(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    stem_verse_mapping_file_paths = ["final/stem_verses_eng.json", "final/stem_verses_nep.json"]  # Replace with the paths to your stem-verse mapping files

    stem_verse_mappings = load_stem_verse_mappings(stem_verse_mapping_file_paths)
    verse_index = build_verse_index(stem_verse_mappings)
    vectors = build_vectors(stem_verse_mappings, verse_index)
    
    output_file_vectors = "stem_vectors.json"  # Replace with the desired output file name for stem vectors
    write_to_json(vectors, output_file_vectors)
