import json

# Function to extract dictionaries into a list and find missing keys
def process_json(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    extracted_list = []
    missing_keys = []

    for key in range(451):  # Check keys from 0 to 450
        key_str = str(key)
        if key_str in data:
            word_dict = data[key_str]
            for k,v in word_dict.items():
                v['forms'] = [k]
                extracted_list.append(v)
            # Add the original key as a word in the list under the 'forms' subkey
            #word_dict['forms'] = [key_str]
            #extracted_list.append(word_dict)
        else:
            #print("HOUSTON WE HAVE A PROBLEM!")
            missing_keys.append(key_str)

    return extracted_list, missing_keys

# Example JSON file path
json_file_path = 'final/My_LLM_Results_batched.json'

# Process the JSON file
extracted_data, missing_keys = process_json(json_file_path)

# Write extracted dictionaries into a list
with open('formatted_data_batch_llm.json', 'w', encoding='utf-8') as file:
    json.dump(extracted_data, file, ensure_ascii=False, indent=2)

# Print missing keys
print("Missing keys between 0 and 450:", missing_keys)
