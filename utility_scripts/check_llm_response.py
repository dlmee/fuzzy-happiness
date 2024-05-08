import json
import json

# Function to calculate the averaged result
def calculate_averaged_result(data):
    total_matches = 0
    total_entries = 0
    
    for word, word_data in data.items():
        if 'stems' in word_data:
            stems_data = word_data['stems']
            total_entries += len(stems_data)
            for stem_data in stems_data:
                stem = stem_data['stem']
                for inner_word, inner_word_data in stem_data.items():
                    if inner_word_data == stem:
                        total_matches += 1
                        break  # Break to count only once per outer word
            
        elif 'stem' in word_data:
            stem = word_data['stem']
            for inner_word, inner_word_data in word_data.items():
                if inner_word_data == stem:
                    total_matches += 1
                    break  # Break to count only once per outer word
            total_entries += 1
        else:
            total_entries +=len(word_data)
    
    if total_entries == 0:
        return 0  # Avoid division by zero
    return total_matches / total_entries

# Path to the JSON file
json_file_path = 'My_LLM_Results.json'

# Read data from JSON file
with open(json_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Calculate the averaged result
averaged_result = calculate_averaged_result(data)

# Print the averaged result
print("Averaged result:", averaged_result)
