import json

# Function to count forms of each length
def count_forms(json_file):
    # Initialize a dictionary to store the counts
    counts = {i: 0 for i in range(51)}

    # Open the JSON file
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Iterate through the keys in the JSON data
    for key, value in data.items():
        # Skip the 'length' key
        if key == '**length**':
            continue

        # Check the length of the 'forms' subkey
        forms_length = len(value.get('forms', []))

        # Increment the count for the corresponding length
        if forms_length <= 50:
            counts[forms_length] += 1

    return counts

# Example usage
json_file = 'new_dictionary_afx.json'  # Replace 'example.json' with the path to your JSON file
result = count_forms(json_file)
print(result)
