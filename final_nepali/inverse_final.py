import json

def inverse_json(input_json):
    with open(input_json, 'r', encoding='utf-8') as file:
        data = json.load(file)
        
    inverted_data = {}
    for stem, stem_data in data.items():
        if stem in ('stem_count', 'word_count'):
            continue
        forms = stem_data['forms']
        definition = stem_data['definition']
        gold = stem_data.get('gold_def', 'UNKNOWN')
        for form in forms:
            inverted_data[form] = {
                'stem': stem,
                'definition': definition,
                'gold': gold
            }
    
    return inverted_data

def main():
    input_json = "swahili/formatted_llm_response_ALL.json"  # Update with your JSON file name
    inverted_data = inverse_json(input_json)
    with open("swahili/inverse_final_swahili.json", "w") as outj:
        json.dump(inverted_data, outj, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
