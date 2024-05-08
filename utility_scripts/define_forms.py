import json

# Function to load Nepali definitions from JSON file
def load_nepali_definitions(json_file):
    nepali_definitions = {}
    with open(json_file, 'r', encoding='utf-8') as file:
        nepali_definitions = json.load(file)
    return nepali_definitions

# Function to attach definitions to Nepali words
def attach_definitions(forms_data, nepali_definitions):
    for word, forms_info in forms_data.items():
        if word == "**length**": continue
        if type(forms_info['definitions']) == str:
            if forms_info['definitions'] == "UNKNOWN":
                definitions = ['NO DICTIONARY DEFINITION']
            else:
                definitions = [forms_info['definitions']]
        else:
            definitions = [defin for defin in forms_info['definitions']]
        for form in forms_info['forms']:
            if form in nepali_definitions:
                definitions.append((form, nepali_definitions[form]))
        forms_info['definitions'] = definitions
    return forms_data

# Path to the Nepali definitions JSON file
nepali_definitions_file = 'nepali_definitions.json'
# Path to the JSON file containing word forms
word_forms_file = 'new_dictionary_afx.json'

# Load Nepali definitions
nepali_definitions = load_nepali_definitions(nepali_definitions_file)

# Load word forms data
with open(word_forms_file, 'r', encoding='utf-8') as file:
    word_forms_data = json.load(file)

# Attach definitions to Nepali words
word_forms_data_with_definitions = attach_definitions(word_forms_data, nepali_definitions)

# Print or use the updated word forms data with definitions
with open("new_dictionary_afx_wdefs.json", "w") as outj:
    json.dump(word_forms_data_with_definitions, outj, ensure_ascii=False, indent=4)
