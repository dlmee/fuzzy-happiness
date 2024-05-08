import json

with open("data/eng_nep_dict.tsv", "r") as inj:
    data = inj.readlines()

# Assuming each line in the TSV has at least two columns and you want the second column as the key and the last column as the value
mydict = {elements[1]: elements[-1].strip() for line in data if len(line.split('\t')) > 1 for elements in [line.split('\t')]}

with open("nep_wdef.json", "w") as outj:
    json.dump(mydict, outj, indent=4, ensure_ascii=False)
