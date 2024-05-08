import json

# Function to extract Nepali words and their definitions
def extract_nepali_definitions(json_file):
    nepali_definitions = {}
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
        for word, info in data.items():
            if isinstance(info, dict):
                for key, value in info.items():
                    for k2, v2 in value.items():
                        if isinstance(v2, dict) and 'definition' in v2:
                            nepali_word = k2.strip()  # Remove any leading/trailing whitespace
                            definition = v2['definition'].strip()  # Remove any leading/trailing whitespace
                            nepali_definitions[nepali_word] = definition
    return nepali_definitions

# Path to the input JSON file
input_json_file = 'data/nepali_purification.json'

# Extract Nepali words and their definitions
nepali_definitions = extract_nepali_definitions(input_json_file)

# Write the extracted data to a new JSON file
output_json_file = 'nepali_definitions.json'
with open(output_json_file, 'w', encoding='utf-8') as file:
    json.dump(nepali_definitions, file, ensure_ascii=False, indent=4)

print("Nepali words and their definitions have been successfully extracted and saved to", output_json_file)
