import csv
import json

def process_csv(csv_file):
    nepali_dict = {}
    with open(csv_file, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            english_word = row[0].strip()
            part_of_speech = row[1].strip()
            nepali_words = [word.strip() for word in row[2].strip('"').split(',')]
            for word in nepali_words:
                if word not in nepali_dict:
                    nepali_dict[word] = [english_word]
                else:
                    nepali_dict[word].append(english_word)
    return nepali_dict

def process_json(json_file, nepali_dict):
    new_entries = 0
    augmented_entries = 0
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
        
    for key in nepali_dict:
        cleaned_key = key.strip('"')  # Remove leading and trailing quotes
        if cleaned_key in data:
            existing_definitions = [data[cleaned_key]]
            for english_word in nepali_dict[key]:
                if english_word not in existing_definitions:
                    existing_definitions.append(english_word)
                    augmented_entries += 1
        else:
            data[cleaned_key] = [word for word in nepali_dict[key]]
            new_entries += 1
    for k,v in data.items():
        if type(v) == str:
            data[k] = [v]
            
    return data, new_entries, augmented_entries


def main():
    csv_file = "data/nep_eng_dict_3.csv"  # Update with your CSV file name
    nepali_dict = process_csv(csv_file)
    json_file = "data/nep_dict_unified_wdef.json"  # Update with your JSON file name
    updated_data, new_entries, augmented_entries = process_json(json_file, nepali_dict)
    print("New entries added:", new_entries)
    print("Existing entries augmented:", augmented_entries)
    #print("Updated JSON data:", updated_data)
    with open("nep3dict_unified_wdef.json", "w") as outj:
        json.dump(updated_data, outj, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
