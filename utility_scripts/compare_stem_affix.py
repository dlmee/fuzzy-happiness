import json

def analyze_stem_affix_overlap(stems_dict, affixes_dict):
    """Analyzes overlap between stem and affix dictionaries, reporting potential misclassifications."""
    output_dict = {}
    
    # Combine prefix and suffix counts into one dictionary for easier lookup
    combined_affixes = affixes_dict['prefixes'].copy()
    combined_affixes.update(affixes_dict['suffixes'])
    
    # Iterate over each stem in the stems dictionary
    for stem, data in stems_dict.items():
        if stem in combined_affixes:
            # Count the number of forms with length greater than 1
            forms_count = len([form for form in data['forms'] if type(form) == list])
            
            # Build the output entry
            output_dict[stem] = {
                'forms_count': forms_count,
                'affix_count': combined_affixes[stem]
            }
    
    return output_dict

with open("new_dictionary.json","r") as inj:
    stems_dict = json.load(inj)
with open("data/affix_counts.json", "r") as inj:
    affixes_dict = json.load(inj)

result = analyze_stem_affix_overlap(stems_dict, affixes_dict)
with open("stem_afx_res.json", "w") as outj:
    json.dump(result, outj, indent=4, ensure_ascii=False)
