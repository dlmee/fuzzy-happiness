import json

def merge_and_process_jsons(json1_path, json2_path, json3_path, json4_path):
    # Load data from the first two JSON files
    with open(json1_path, 'r', encoding='utf-8') as file:
        data1 = json.load(file)
    
    with open(json2_path, 'r', encoding='utf-8') as file:
        data2 = json.load(file)
    
    # Load data from the third JSON file
    with open(json3_path, 'r', encoding='utf-8') as file:
        data3 = json.load(file)

    with open(json4_path, 'r', encoding='utf-8') as file:
        data4 = json.load(file)
    
    # Create a final dictionary to store the merged and processed data
    final_data = {}
    
    # Merge data from the first two JSON files
    for entry in data1 + data2:
        if entry['proper noun'] == 'yes':
            continue
        stem = entry['stem']
        if stem not in final_data:
            final_data[stem] = {
                'definition': [entry['definition']],
                'forms': entry['forms']
            }
        else:
            final_data[stem]['definition'].append(entry['definition'])
            final_data[stem]['forms'] += entry['forms']
    
    # Process definitions: merge them into a single string and eliminate duplicates
    for stem, value in final_data.items():
        merged_definitions = set()
        for definition in value['definition']:
            merged_definitions.update(definition.split(', '))
        value['definition'] = ', '.join(merged_definitions)
    
    # Check data against the third JSON file
    for stem, value in final_data.items():
        if stem in data3:
            value['gold_def'] = data3[stem].get('definitions', ["UNKNOWN"])
        elif stem in data4:
            value['gold_def'] = data4[stem]
        else:
            fmatch = False
            for form in value['forms']:
                if form in data4:
                    value['gold_def'] = data4[form]
                    fmatch=True
            if not fmatch:
                value['gold_def'] = ["UNKNOWN"]
    
    # Filter out entries with proper noun 'yes'
    print(f"The length of final_data is {len(final_data)}")
    final_data = {stem: value for stem, value in final_data.items() if value.get('proper noun') != 'yes'}
    print(f"The length of final_data is {len(final_data)}")
    # Calculate stem count and word count
    stem_count = len(final_data)
    word_count = sum(len(value['forms']) for value in final_data.values())
    
    # Add stem count and word count at the top of the dictionary
    final_data = {'stem_count': stem_count, 'word_count': word_count, **final_data}
    
    # Sort the dictionary by length of forms (except stem count and word count keys)
    sorted_data = sorted(
        [(k, v) for k, v in final_data.items() if k not in ('stem_count', 'word_count')],
        key=lambda item: len(item[1]['forms']),
        reverse=True
    )
    
    # Reconstruct the dictionary with sorted data
    final_data = {'stem_count': final_data['stem_count'], 'word_count': final_data['word_count']}
    for k, v in sorted_data:
        final_data[k] = v
    
    return final_data

# Example usage:
json1_path = 'final/formatted_data_batch_llm.json'
json2_path = 'final/formatted_llm_response.json'
json3_path = 'new_dictionary_afx_final.json'
json4_path = 'data/nep3dict_unified_wdef.json'

final_data = merge_and_process_jsons(json1_path, json2_path, json3_path, json4_path)

# Print the final data
with open("final/final_nepali.json", "w") as outj:
    json.dump(final_data, outj, indent=4, ensure_ascii=False)
