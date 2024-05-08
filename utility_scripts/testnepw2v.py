import json
import numpy as np
import random
from tqdm import tqdm

# Function to load word embeddings from a word2vec text file
def load_word2vec(file_path, word_targets):
    word_embeddings = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in tqdm(file):
            values = line.strip().split()
            word = values[0]
            #if word not in word_targets: continue
            embedding = np.array(values[1:], dtype=np.float32).astype(float)
            word_embeddings.append(word)
    return word_embeddings

# Function to load word forms and stem from a JSON file

def load_word_forms(json_file):
    word_forms = {}
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
        for word, info in data.items():
            if word != '**length**':
                forms = info.get('forms', [])
                stem = info.get('stem', '')
                word_forms[word] = {'forms': forms, 'stem': stem}
    return word_forms

# Example usage
json_file = 'new_dictionary_afx.json'
word_forms = load_word_forms(json_file)
# Now you have a dictionary where keys are words, and values are dictionaries containing forms and stem


# Example usage
word2vec_file = 'data/nepali_embeddings_word2vec.txt'
word_embeddings = load_word2vec(word2vec_file, set(word_forms))
with open('data/w2vnep_allkeys.json', 'w') as outj:
    json.dump(word_embeddings, outj, indent=4, ensure_ascii=False)

# Now you have a dictionary where keys are words and values are their respective embeddings



# Function to load word embeddings from a JSON file
def load_word_embeddings(json_file):
    with open(json_file, 'r') as inj:
        return json.load(inj)

# Function to calculate cosine similarity between two words
def cosine_similarity(word1, word2, embeddings):
    if word1 not in embeddings or word2 not in embeddings:
        return "Word not found in embeddings"
    else:
        embedding1 = embeddings[word1]
        embedding2 = embeddings[word2]
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        similarity = dot_product / (norm1 * norm2)
        return similarity



# Example usage
json_file = 'data/w2vnep.json'
word_embeddings = load_word_embeddings(json_file)

# Specify words to compare
all_comparisons_by_length = {'holistic':{'key_hits': [], 'word_key_hits': [], 'avg_dist': []}}

for k, v in word_forms.items():
    word_length = len(v['forms'])
    if k in word_embeddings:
        if word_length not in all_comparisons_by_length:
            all_comparisons_by_length[word_length] = {'key_hits': [], 'word_key_hits': [], 'avg_dist': []}
        
        all_comparisons_by_length[word_length]['key_hits'].append(1)
        all_comparisons_by_length['holistic']['key_hits'].append(1)
        
        targets = v['forms']
        match = False
        for word in targets:
            if word in word_embeddings:
                all_comparisons_by_length[word_length]['word_key_hits'].append(1)
                all_comparisons_by_length['holistic']['word_key_hits'].append(1)
                all_comparisons_by_length[word_length]['avg_dist'].append(cosine_similarity(k, word, word_embeddings))
                all_comparisons_by_length['holistic']['avg_dist'].append(cosine_similarity(k, word, word_embeddings))
                match = True
        
        if not match:
            all_comparisons_by_length[word_length]['word_key_hits'].append(0)
            all_comparisons_by_length['holistic']['word_key_hits'].append(0)
    else:
        if word_length not in all_comparisons_by_length:
            all_comparisons_by_length[word_length] = {'key_hits': [], 'word_key_hits': [], 'avg_dist': []}
        
        all_comparisons_by_length[word_length]['key_hits'].append(0)
        all_comparisons_by_length['holistic']['key_hits'].append(0)
        #all_comparisons_by_length[word_length]['word_key_hits'].append(0)




for k,v in all_comparisons_by_length.items():
    for k2, v2 in v.items():
        if v2:
            mylength = len(v2)
            all_comparisons_by_length[k][k2] = sum(v2)/len(v2)
    all_comparisons_by_length[k]['number of forms'] = mylength
with open("w2vecvalidation.json", "w") as outj:
    json.dump(all_comparisons_by_length, outj, indent=4, ensure_ascii=False)
