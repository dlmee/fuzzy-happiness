import json
import json

# Function to calculate the averaged result
def calculate_averaged_result(data):
    total_matches = 0
    total_entries = 0
    formatted = {}
    for word, word_data in data.items():
        if 'stems' in word_data:
            stems_data = word_data['stems']
            total_entries += len(stems_data)
            for stem_data in stems_data:
                if stem_data['proper noun'] == "yes": continue
                correct_stem = stem_data['stem']
                forms = stem_data['forms']
                definition = stem_data['definition']
                if correct_stem in formatted:
                    formatted[correct_stem]['forms'] = formatted[correct_stem]['forms'] + forms
                    formatted[correct_stem]['definition'] =  formatted[correct_stem]['definition'] + [definition]
                else:
                    formatted[correct_stem] =  {'forms':forms, 'definition':[definition]}
                stem = stem_data['stem']
                for inner_word, inner_word_data in stem_data.items():
                    if inner_word_data == stem:
                        total_matches += 1
                        break  # Break to count only once per outer word
            
        elif 'stem' in word_data:
            stem = word_data['stem']
            stem_data = word_data
            if stem_data['proper noun'] == "yes": continue
            correct_stem = stem_data['stem']
            forms = stem_data['forms']
            definition = stem_data['definition']
            if correct_stem in formatted:
                formatted[correct_stem]['forms'] = formatted[correct_stem]['forms'] + forms
                formatted[correct_stem]['definition'] =  formatted[correct_stem]['definition'] + [definition]
            else:
                formatted[correct_stem] =  {'forms':forms, 'definition':[definition]}
            for inner_word, inner_word_data in word_data.items():
                if inner_word_data == stem:
                    total_matches += 1
                    break  # Break to count only once per outer word
            total_entries += 1
        else:
            total_entries +=len(word_data)
            for k,v in word_data.items():
                stem_data = v
                if stem_data['proper noun'] == "yes": continue
                correct_stem = stem_data['stem']
                forms = stem_data['forms']
                definition = stem_data['definition']
                if correct_stem in formatted:
                    formatted[correct_stem]['forms'] = formatted[correct_stem]['forms'] + forms
                    formatted[correct_stem]['definition'] =  formatted[correct_stem]['definition'] + [definition]
                else:
                    formatted[correct_stem] =  {'forms':forms, 'definition':[definition]}
    
    if total_entries == 0:
        return 0  # Avoid division by zero
    print(f"The length of formatted is {len(formatted)}")
    print(f"The length of formatted is {sum([len(v['forms']) for k,v in formatted.items()])}")
    return total_matches / total_entries, formatted

def calculate_multiple_result(data):
    average, formatted = calculate_averaged_result(data[0])
    print("and done!")
    average2 = []
    validation = []
    for k,v in data[1].items():
        validation.append(int(k))
        for k2, v2 in v.items():
            if k2 == v2['stem']:
                average2.append(1)
            else:
                average2.append(0)
            if v2['proper noun'] == 'yes' or v2['stopword'] == 'yes': continue
            correct_stem = v2['stem']
            forms = [k2]
            definition = v2['definition']
            if correct_stem in formatted:
                    formatted[correct_stem]['forms'] = formatted[correct_stem]['forms'] + forms
                    formatted[correct_stem]['definition'] =  formatted[correct_stem]['definition'] + [definition]
            else:
                formatted[correct_stem] =  {'forms':forms, 'definition':[definition]}
    
    print(f"The length of formatted is now {len(formatted)}")
    print(f"The length of formatted is now {sum([len(v['forms']) for k,v in formatted.items()])}")        
    return (average + (sum(average2)/len(average2)))/2, formatted


# Path to the JSON file
json_file_path = 'swahili/My_LLM_Results_stem_validation.json'
json_file_path_2 = 'swahili/My_LLM_Results_stem_validation_batched.json'

# Read data from JSON file
with open(json_file_path, 'r', encoding='utf-8') as file:
    data = [json.load(file)]
with open(json_file_path_2, 'r', encoding='utf-8') as file:
    data.append(json.load(file))

# Calculate the averaged result
if len(data) == 1:
    averaged_result, formatted = calculate_averaged_result(data)
else:
    averaged_result, formatted = calculate_multiple_result(data)

mydictionary = 'swahili/sw_dict_unified_wdef.json'

if mydictionary:
    with open(mydictionary, "r") as inj:
        mydictionary = json.load(inj)
    for k,v in formatted.items():
        v['definition'] = list(set(v['definition']))
        if k in mydictionary:
            v['gold_def'] = mydictionary[k]
            wmatch = True
        else:
            wmatch = False
            for word in v['forms']:
                if word in mydictionary:
                    v['gold_def'] = mydictionary[word]
                    wmatch= True
                    break
        if not wmatch:
            v['gold_def'] = ["UNKNOWN"]
    print("And done!")

with open("formatted_llm_response_ALL.json", "w") as outj:
    json.dump(formatted, outj, indent=4, ensure_ascii=False)
# Print the averaged result
print("Averaged result:", averaged_result)
