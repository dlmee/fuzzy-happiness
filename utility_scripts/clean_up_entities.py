import json

def read_json(filename):
    """Read a JSON file and return the data."""
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def extract_capitalized_entities(data):
    """Extract only Nepali words whose corresponding English words are capitalized."""
    extracted_entities = []
    for book, content in data.items():
        named_entities = content.get('named entities', {})
        for nepali, english in named_entities.items():
            if english[0].isupper():
                if " " in nepali:
                    nepali = nepali.split()
                    for n in nepali:
                        extracted_entities.append(n)  # Check if the English word is capitalized
                else:
                    extracted_entities.append(nepali)
    return extracted_entities

def write_json(data, filename):
    """Write data to a JSON file."""
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def process_json(input_file, output_file):
    """Process the JSON file to extract and write out the required entities."""
    data = read_json(input_file)
    extracted_entities = extract_capitalized_entities(data)
    write_json(extracted_entities, output_file)

# Usage
input_file = 'data/nepali_entities.json'  # Replace with the path to your JSON file
output_file = 'data/extracted_entities.json'
process_json(input_file, output_file)
