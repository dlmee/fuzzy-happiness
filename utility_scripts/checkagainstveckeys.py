import json

def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def check_forms(json1, json2):
    forms_set = set(json2)
    key_match = []
    form_match = []
    absent_keys = []
    for key, value in json1.items():
        kmatch = False
        fmatch = False
        if key in ["stem_count", "word_count"]: continue
        if key in forms_set:
            key_match.append(1)
            kmatch = True
        if 'forms' in value:
            for form in value['forms']:
                if form in forms_set:
                    form_match.append(1)
                    fmatch = True
                else:
                    form_match.append(0)
        if not kmatch and not fmatch:
            key_match.append(0)
            absent_keys.append([key, len(value['forms'])])

    return (sum(key_match)/len(key_match), sum(form_match)/len(form_match)), absent_keys

if __name__ == "__main__":
    json1_file = "final/final_nepali.json"
    json2_file = "data/w2vnep_allkeys.json"
    
    json1 = load_json(json1_file)
    json2 = load_json(json2_file)
    
    results, absent_keys = check_forms(json1, json2)
    print(results)
    with open("absent_keys.json", "w") as outj:
        json.dump(absent_keys, outj, indent=4, ensure_ascii=False)
