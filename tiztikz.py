import re
import json
import math
from tqdm import tqdm
import random
import copy
import itertools
from itertools import permutations
import numpy as np
from Levenshtein import distance as lstn

class Tiztikz:
    def __init__(self, corpus, grammar = None, allgrams = None, stems=None, entities=None):
        tokens, counts = self.read_dev_txt(corpus)
        self.allgrams = self.morph_merge(tokens, counts, grammar=grammar, allgrams=allgrams, stems=stems, entities=entities)
        with open("morphemic_breakdown.json", "w") as outj:
            json.dump(self.allgrams, outj, indent=4, ensure_ascii=False)
        

    def read_txt(self, corpus):
        text = corpus
        with open(text, encoding="utf-8") as fi:
            lines = fi.read()
        lines = re.split("[\.\?\!]", lines)
        lines = [[re.sub("[^a-z]", "", word.lower()) for word in re.split(" |-|\n|—", line)] for line in lines]
        lines = [[word for word in line if word] for line in lines]
        return lines

    def read_dev_txt(self, corpus):
        text = corpus
        with open(text, encoding="utf-8") as fi:
            lines = fi.read()
        # Split lines based on sentence end markers
        lines = re.split("[\.\?\!]", lines)
        # Define a pattern to match Devanagari characters only
        devanagari_pattern = r'[\u0900-\u097F]+'
        # Process each line
        lines = [[re.sub(r"[^\u0900-\u097F]", "", word) for word in re.split(" |-|\n|—", line)] for line in lines]
        # Filter out empty words and words containing Roman characters
        counts = {'**total**': 0}
        for line in lines:
            for word in line:
                if not word: continue
                if word not in counts:
                    counts[word] = 1
                    counts['**total**'] +=1
                else:
                    counts[word] += 1
                    counts['**total**'] +=1
        for k,v in counts.items():
            if k != '**total**':
                counts[k] = v/counts['**total**']
        counts = sorted(list(counts.items()), key = lambda x:x[1], reverse = True)
        counts = {word[0]:word[1] for word in counts}
        lines = set([word for line in lines for word in line if word and re.fullmatch(devanagari_pattern, word)])
        return list(lines), counts


    def morph_merge(self, tokens, counts, grammar=None, allgrams=None, stems=None, entities=None):
        if allgrams:
            with open(allgrams, 'r') as inj:
                allgrams = json.load(inj)
        else:
            allgrams = self._first_pass(tokens, counts, stems, entities)
            allgrams = self._second_pass(allgrams, tokens)
            with open("allgrams.json", 'w') as outj:
                json.dump(allgrams, outj, indent=4, ensure_ascii=False)
            
        allgrams = self._third_pass(allgrams, tokens, stems, grammar=grammar)
        return allgrams

    def _first_pass(self, tokens, counts, stems, entities):
        #Need to collect the types of words, to know what we have to be able to build. 
        allgrams = {'**word**': {'**total**': {}}}
        if stems:
            stem_dict = self.side_hustle(tokens, stems, entities, counts)
            print("and done!")
            for k,v in tqdm(stem_dict.items(), desc="Adding stems to allgrams"):
                if k == 'non-fitting':
                    for word in v:
                        allgrams = self._add_grams(word, allgrams)
                else:
                    for word in v:
                        chunks = word.split(k)
                        for chunk in chunks:
                            allgrams = self._add_grams(chunk, allgrams)
                        allgrams = self._add_grams(word, allgrams, stem=True)

        else:
            for word in tqdm(tokens, desc='First Pass'):
                self._add_grams(word, allgrams)
                
        return allgrams

    def _add_grams(self, word, allgrams, stem=False):
        if stem:
            thisgrams = {len(word):[word]}
        else:
            thisgrams = self._find_grams(word)
        for n, grams in thisgrams.items():
            if n not in allgrams:
                allgrams[n] = {"**total**": 0, '**threshold**': 0}
            for gram in grams:
                allgrams[n]["**total**"] += 1
                if gram not in allgrams[n]:
                    allgrams[n][gram] = {'count': 1, 'prep': 0, 'sufp': 0, 'dictionary':stem, 'before': {'#':0}, 'after': {'#':0}, 'coverage': [word]}
                else:
                    allgrams[n][gram]['count'] += 1
                    if word not in allgrams[n][gram]['coverage']:
                        allgrams[n][gram]['coverage'].append(word)
        return allgrams


    def _second_pass(self, allgrams, tokens):
        for word in tqdm(tokens, desc="Second pass"):
            if not word: continue
            if len(word) > 14: continue
            possibles = self.generate_splits_iterative(word)
            # = self.generate_splits_components(word)
            for stem, grams in possibles.items():
                n = len(stem)
                if stem not in allgrams[n]: continue
                for key in ['before', 'after']:
                    for gram in list(grams[key]):
                        if gram not in allgrams[n][stem][key]:
                            allgrams[n][stem][key][gram] = 1
                        else:
                            allgrams[n][stem][key][gram] +=1

                allgrams[n][stem]['prep'] = allgrams[n][stem]['before']['#'] / (sum([count for key, count in allgrams[n][stem]['before'].items() if key != '#']) +1)
                allgrams[n][stem]['sufp'] = allgrams[n][stem]['after']['#'] / (sum([count for key, count in allgrams[n][stem]['after'].items() if key != '#']) + 1)
        return allgrams

    def _third_pass(self, allgrams, tokens, stems = None, grammar = None):
        allgrams, stems = self.transform_allgrams(allgrams, stems)
        #self.side_hustle(tokens, stems)
        #Here is where I need to start building rules
        if grammar:
            with open(grammar, 'r') as inj:
                grammar = json.load(inj)
                efficiency = grammar['cost']
                probability = grammar['probability']
                grammars = [grammar]
        else:
            if not stems:
                stems =[(k,0) for k,v in allgrams.items() if v['stems'] == 0 and len(k) > 2]
            else:
                stems =[pair for pair in stems if len(pair[0]) > 1]
            grammar = {
        'probability': float('-inf'),
        'cost': float('inf'),
        'stems': stems,
        'prefixes':[('NULL', 0)] + allgrams['prefixes'][100:500],
        'suffixes':[('NULL', 0)] + allgrams['suffixes'][100:500],
        'combinations': []
    }

            #let's initialized an unprobable and expensive grammar, every word and no affixes or stems!
            grammars = [self.build_grammar_stem(grammar, tokens, allgrams)]
            efficiency = grammar['cost']
            probability = float('-inf')
        c_improved = True
        t = 100
        counter = 0
        growing = ['grow','shrink', 'mutate']
        while t > 10:
            grammars = sorted(grammars, key= lambda x:x['cost'])
            grammars = grammars[:1]
            with open(f"new_best_grammar.json", "w") as outj:
                json.dump(grammars[0], outj, indent=4, ensure_ascii=False)
            newgrammars, allgrams = self.ratchet_grammar(grammars, tokens, allgrams, temp=random.randint(1,t), growing=growing, paths=3)
            print(f"The length of allgrams stems is: {len(allgrams['stems'])}. The length of allgrams prefixes is: {len(allgrams['prefixes'])}. The length of allgrams suffixes is: {len(allgrams['suffixes'])}")
            previous_grammars = grammars
            grammars = []
            c_improved = False
            p_improved = False
            for k,v in newgrammars.items():
                print(f"The cost of {k} = {v['cost']}, prob = {v['probability']}")
                if v['cost'] <= efficiency:
                    efficiency = v['cost']
                    grammars.append(v)
                    c_improved = True
                    t += 10
                    growing.append(random.choice(['grow','grow','grow','grow','shrink','mutate']))
                elif v['probability'] > probability:
                    probability = v['probability']
                    p_improved = True
                    grammars.append(v)
            if not c_improved:
                t -= 20
                growing.append(random.choice(['shrink','mutate']))
                grammars.append(previous_grammars[0])
            """if not p_improved:
                t -= 20
                growing.append(random.choice(['shrink','mutate']))
                grammars.append(previous_grammars[-1])"""
            print(f"Cost is {c_improved}, and Probability is {p_improved}")
        return grammars[0]

    
    def build_grammar_stem(self, grammar, tokens, allgrams):
        #Need to make sure that our rules are building correctly, but we now have the probabilities squared away!
        grammar['combinations'] = []
        for word in tqdm(tokens, desc="building grammar"):
            if str(len(word)) in allgrams:
                if word not in allgrams[str(len(word))]:
                    continue
            decomposition = self.recursive_decompose(word, [pre for pre, prob in grammar['prefixes']], [stem for stem, prob in grammar['stems']], [suf for suf, prob in grammar['suffixes']])
            if decomposition:
                breakdown = [(el, allgrams[str(len(el))][el][k]) for k,v in decomposition.items() for el in v]
                word = '-'.join([br[0] for br in breakdown])
                prob = sum([br[1] for br in breakdown])/len(breakdown)
                if prob == float('-inf'):
                    prob = [-50]
                #decomposition['stems'] = decomposition['stems']
                grammar = self.shift(grammar, decomposition) #This makes used rules less likely to be dropped. 
                stempool = [s for s in decomposition['stems']]
                if grammar['combinations']:
                    for stem in stempool:
                        for affixation in grammar['combinations']:
                            if affixation['stems']:
                                if stem == affixation['stems']:
                                    for prfx in decomposition['prefixes']:
                                        if prfx not in affixation['prefixes']:
                                            affixation['prefixes'].append(prfx)
                                    for sffx in decomposition['suffixes']:
                                        if sffx not in affixation['suffixes']:
                                            affixation['suffixes'].append(sffx)
                                    affixation['generates'].append((word))
                                    affixation['probability'].append(prob)
                                    stempool.remove(stem)
                                    break
                if stempool: #didn't find a matching rule, need to make a rule
                    for stem in stempool:
                        grammar['combinations'].append({
                            'stems': stem,
                            'prefixes': decomposition['prefixes'],
                            'suffixes': decomposition['suffixes'],
                            'probability': [prob], #we'll need to build a custom function here :D 
                            'generates': [word]
                        })
            else:
                if word in allgrams[str(len(word))]:
                    probability = [allgrams[str(len(word))][word]['stems']]
                else:
                    probability = [-50]
                grammar['combinations'].append(
                    {
            'stems': [word],
            'prefixes': None,
            'suffixes': None,
            'probability': probability,
            'generates': [word]
        })
        grammar['probability'] = sum(p for c in grammar['combinations'] for p in c['probability'])
        #grammar['cost'] = sum([len(v['generates']) for v in grammar['combinations']])/len(grammar['combinations'])
        grammar['cost'] = sum([abs(len(v['generates']) - 25) for v in grammar['combinations']]) / len(grammar['combinations'])
        grammar['combinations'] = sorted(grammar['combinations'], key = lambda x:len(x['generates']), reverse=True)
        return grammar
    
    def shift(self, grammar, decomposition):
        for k,v in decomposition.items():
            for word in v:
                if len(word) == 1: continue
                if word in grammar[k]:
                    grammar[k].insert(0, grammar[k].pop(grammar[k].index(word)))
        return grammar
        
    
    def recursive_decompose(self, target_word, prefixes, stems, suffixes, found_components=None, depth=0):
        if found_components is None:
            found_components = {'prefixes': [], 'stems': [], 'suffixes': []}

        # Base case: If the target word matches any stem directly or no more target word to check
        if target_word in stems:
            if not found_components['prefixes'] and not found_components['suffixes']:
                return None
            found_components['stems'] = [target_word] #.append(target_word)
            #print(f"Final decomposition at depth {depth}: {found_components}")
            return found_components
        elif not target_word or depth > 10:  # Prevent infinite recursion
            #print(f"No decomposition found for '{target_word}' at depth {depth}")
            return None

        for suffix in suffixes:
            if target_word.endswith(suffix):
                # Update target_word by removing the suffix
                new_target = target_word[:-len(suffix)]
                found_components['suffixes'].append(suffix)

                # Recursive call
                return self.recursive_decompose(new_target, prefixes, stems, suffixes, found_components, depth + 1)
        
        for prefix in prefixes:
            if target_word.startswith(prefix):
                # Update target_word by removing the prefix
                new_target = target_word[len(prefix):]
                found_components['prefixes'].append(prefix)

                # Recursive call
                return self.recursive_decompose(new_target, prefixes, stems, suffixes, found_components, depth + 1)

        # If no prefix or suffix matches, try deeper decomposition if not solely relying on the stem
        if depth > 0:
            pass
            #print(f"Reached a decomposable state at depth {depth} without isolating a stem: {found_components}")
        else:
            #print(f"No decomposition possible for '{target_word}'")
            pass
        return None


    def ratchet_grammar(self, grammars, tokens, allgrams, temp=1, paths=3, growing=['grow']):
        def mutate(mutation, allgrams, targets, temp):
            for k,v in targets.items():
                if len(mutation[k]) < v: continue
                average = mutation['cost']
                for _ in range(v):
                    element = random.choice(mutation[k])
                    if element[1] == 0: continue #This is to prevent leaking dictionary stems.
                    if k == 'stems':
                        count = sum([len(v2['generates']) for v2 in mutation['combinations'] if element[0] in v2[k]])
                    else:
                        count = sum([len(v2['generates']) for v2 in mutation['combinations'] if element[0] == v2[k]])
                    if count > average:
                        combinations = [pair for pair in allgrams[k] if lstn(element[0],pair[0]) <= 2 and pair[1] > element[1]]
                        if combinations:
                            replacement = random.choice(combinations)
                            print(f"mutating {k}")
                            mutation[k].remove(element)
                            mutation[k].append(replacement)
                        if k in ['prefixes', 'suffixes']:
                            if ('NULL', 0) not in mutation[k]:
                                mutation[k].append(('NULL',0))
            return mutation
        
        def grow(additive, allgrams, targets, temp):
            for k,v in targets.items():
                for _ in range((v+1)):
                    addition, allgrams = self.suggest_element(allgrams, additive[k], allgrams[k], temp, k)
                    additive[k].append(addition)
            return additive, allgrams

        def shrink(purge, allgrams, targets, temp):
            for k,v in targets.items():
                counter = 0
                #if len(purge[k]) < v: continue
                average = purge['cost'] + 15
                """for _ in range(v):
                    element = random.choice(purge[k])"""
                for element in purge[k]:
                    if element == 'NULL':
                        raise
                    if element[0] == 'NULL': continue
                    if element[1] == 0 and len(element[0]) > 2: continue # prevent leaking dictionary stems
                    if k == 'stems':
                        count = sum([len(v2['generates']) for v2 in purge['combinations'] if element[0] in v2[k]])
                    else:
                        count = sum([len(v2['generates']) for v2 in purge['combinations'] if element[0] == v2[k]])
                    if count > average:
                        purge[k].remove(element)
                        print(f"purging {k}")
                        counter +=1
                    if k in ['prefixes', 'suffixes']:
                        if ('NULL', 0) not in purge[k]:
                            purge[k].append(('NULL', 0))
                    if counter == v:
                        break
            return purge
        
        pgrammars = {}
        for j, grammar in enumerate(grammars):
            for i in range(1, paths + 1):
                growing = ['grow', 'shrink', 'mutate'][i%3]
                targets = {'stems': 2*temp, 'prefixes':1*temp,'suffixes':1*temp}
                mypath = copy.deepcopy(grammar)
                if growing == 'grow':
                    mypath, allgrams = grow(mypath, allgrams, targets, temp)
                elif growing == 'mutate':
                    mypath = mutate(mypath, allgrams, targets, temp)
                elif growing == 'shrink':
                    mypath = shrink(mypath, allgrams, targets, temp)
                    mypath, allgrams = grow(mypath, allgrams, targets, temp)
                pgrammars[f"Possible {j}:{i}"] = self.build_grammar_stem(mypath, tokens, allgrams)
        return pgrammars, allgrams
    
    def suggest_element(self, master, previous, allgrams, temp, k_type):
        #allgrams = list(allgrams.keys())
        if 'used' not in master:
            master['used'] = {}
        for first in range(len(allgrams)-1):
            if allgrams[first] not in previous:
                break
        if first not in master['used']:
            master['used'][first] = 1
        else:
            master['used'][first] += 1
        if master['used'][first] >= len(allgrams[first][0]):
            master[k_type].pop(first)
            #then we want to preclude it from use. 
        return allgrams[first], master



    def _calculate_thresholds(self, v):
        conditionals_lengths = [len(v2['conditionals']) for v2 in v.values() if isinstance(v2, dict)]
        if not conditionals_lengths: return
        total_avg = sum(conditionals_lengths) / len(conditionals_lengths)
        total_max = max(conditionals_lengths)
        total_min = min(conditionals_lengths)
        if total_max == total_min:
            v['**threshold**'] = 0
        else:
            v['**threshold**'] = (total_avg - total_min) / (total_max - total_min)
        for k2, v2 in v.items():
            if k2 in ["**total**", "**threshold**"]: continue
            if total_max == total_min:
                v2['range'] = 1
            else:
                v2['range'] = (len(v2['conditionals']) - total_min) / (total_max - total_min)

    def _find_probabilities(self, tokens, allgrams):
        probabilities = []
        counter = 0
        for line in tokens:
            for word in line:
                counter += 1
                print(self._find_probabilities_for_word(word, allgrams))
                if counter == 50: break
            if counter == 50: break
        return probabilities

    def _find_probabilities_for_word(self, word, allgrams):
        # This method would contain the logic for calculating probabilities for a single word.
        pass


    def generate_splits_iterative(self, word):
        num_positions = len(word) - 1
        num_combinations = 1 << num_positions
        all_chunks = {}

        def add_combinations(segment, target_list):
            # Generate all combinations of the segment by progressively building subsegments
            for start in range(len(segment)):
                for end in range(start + 1, len(segment) + 1):
                    subsegment = segment[start:end]
                    if subsegment not in target_list:
                        target_list.add(subsegment)

        for i in range(num_combinations):
            chunks = [word[0]]  # Start with the first character

            for j in range(num_positions):
                if i & (1 << j):  # If the j-th bit is set, start a new chunk
                    chunks.append('')
                chunks[-1] += word[j + 1]

            for idx, chunk in enumerate(chunks):
                if chunk not in all_chunks:
                    all_chunks[chunk] = {'before': set(), 'after': set()}

                # Compute the before and after parts
                before = ''.join(chunks[:idx]) if idx > 0 else '#'
                after = ''.join(chunks[idx + 1:]) if idx < len(chunks) - 1 else '#'

                # Add combinations of before and after parts
                if before != '#':
                    add_combinations(before, all_chunks[chunk]['before'])
                else:
                    all_chunks[chunk]['before'].add(before)
                    
                if after != '#':
                    add_combinations(after, all_chunks[chunk]['after'])
                else:
                    all_chunks[chunk]['after'].add(after)

        return all_chunks


    def generate_splits_components(self, word):
        all_components = set()  # Use a set to avoid duplicate components

        # Generate all possible substrings (components) of the word
        for i in range(len(word)):
            for j in range(i + 1, len(word) + 1):
                component = word[i:j]
                all_components.add(component)

        return list(all_components)  # Convert the set to a list to return it

    def _find_grams(self, word):
        grams = {len(word):[word]}
        for n in range(1, len(word) + 1):  # Window sizes from 1 to the length of the word
            grams[n] = [word[i:i+n] for i in range(len(word) - n + 1)]
        return grams
    
    def normalize_probabilities(self, transformed):
        # For each category, normalize the probabilities so they sum to 1
        for category in transformed:
            # Calculate the sum of all probabilities in the current category
            total_probability = sum(transformed[category].values())

            # Avoid division by zero
            if total_probability > 0:
                # Normalize each probability by dividing it by the total sum
                for element in transformed[category]:
                    transformed[category][element] /= total_probability

        return transformed

    def normalize_log_probs(self, transformed):
        # For each category, calculate the log probabilities and normalize them
        for category in transformed:
            max_log_prob = max(transformed[category].values())  # This helps to avoid underflow
            total_probability_log = math.log(sum(math.exp(log_prob - max_log_prob) for log_prob in transformed[category].values())) + max_log_prob

            # Normalize each log probability by subtracting the log of the total sum
            for element in transformed[category]:
                transformed[category][element] = transformed[category][element] - total_probability_log

        return transformed


    def safe_log(self, x, y, fallback=-float('inf')):
        """Returns the logarithm of x, or a fallback value if x is 0."""
        if x > 0 and y > 0:
            return math.log(x/y)
        else:
            return fallback


    def transform_allgrams(self, allgrams, stems):
        transformed = {}
        stage_1 = {}
        #Stage 1 range of xx / range all xx for n, types. 
        for k,v in allgrams.items():
            if k in ["**total**", "**threshold**", "**word**"]: continue
            for k2, v2 in v.items():
                if k2 in ["**total**", "**threshold**", "**word**"]: continue
                if len(k2) not in stage_1:
                    stage_1[len(k2)] = {'**total**':0}
                if k2 not in stage_1[len(k2)]:
                    stage_1[len(k2)][k2] = len(v2['coverage'])
                else:
                    stage_1[len(k2)][k2] += len(v2['coverage'])
                stage_1[len(k2)]['**total**'] += len(v2['coverage'])
        print("And done!")

        #Stage 2 targ xx/ all xx for n n3 = xxx etc. tokens. 
        stage_2 = {}
        for k,v in allgrams.items():
            if k in ["**total**", "**threshold**", "**word**"]: continue
            for k2, v2 in v.items():
                if k2 in ["**total**", "**threshold**", "**word**"]: continue
                if len(k2) not in stage_2:
                    stage_2[len(k2)] = {'**total**':0}
                if k2 not in stage_2[len(k2)]:
                    stage_2[len(k2)][k2] = v2['count']
                else:
                    stage_2[len(k2)][k2] += v2['count']
                stage_2[len(k2)]['**total**'] += v2['count']
        print("And done!")

        #Stage 3, conditional, prefix, suffix, neither. 
        stage_3 = {}
        for k,v in allgrams.items():
            if k in ["**total**", "**threshold**", "**word**"]: continue
            for k2, v2 in v.items():
                if k2 in ["**total**", "**threshold**", "**word**"]: continue
                if len(k2) not in stage_3:
                    stage_3[len(k2)] = {'prefixes':0, 'suffixes':0, 'stemix':0}
                if k2 not in stage_3[len(k2)]:
                    stage_3[len(k2)][k2] = {'prefixes':v2['before']['#'], 'suffixes':v2['after']['#'], 'stemix':max([sum([p for k,p in v2['before'].items() if k != '#']) + sum([p for k, p in v2['after'].items() if k != '#']), min(v2['before']['#'], v2['after']['#'])])}
                else:
                    stage_3[len(k2)][k2]['prefixes'] += v2['before']['#']
                    stage_3[len(k2)][k2]['suffixes'] += v2['after']['#']
                    stage_3[len(k2)][k2]['stemix'] += max([(sum([p for k,p in v2['before'].items() if k != '#']) + sum([p for k, p in v2['after'].items() if k != '#'])), min(v2['before']['#'], v2['after']['#'])])
                stage_3[len(k2)]['prefixes'] += v2['before']['#']
                stage_3[len(k2)]['suffixes'] += v2['after']['#']
                stage_3[len(k2)]['stemix'] += max([(sum([p for k,p in v2['before'].items() if k != '#']) + sum([p for k, p in v2['after'].items() if k != '#'])), min(v2['before']['#'], v2['after']['#'])])
        print("And done!")
        
        #Just need to fix the stems
        transformed = {str(k):{} for k in stage_1.keys()}
        for n, details in allgrams.items():
            for element, data in details.items():
                if not element: continue
                if element in ["**total**", "**threshold**"]:  # Skip meta keys
                    continue
                if element not in transformed[n]:
                    s1_lognorm = self.safe_log(stage_1[len(element)][element],stage_1[len(element)]['**total**'])
                    s2_lognorm = self.safe_log(stage_2[len(element)][element],stage_2[len(element)]['**total**'])
                    transformed[n][element] = {
                        "prefixes": s1_lognorm + s2_lognorm + self.safe_log(stage_3[len(element)][element]['prefixes'],stage_3[len(element)]['prefixes']),
                        "suffixes": s1_lognorm + s2_lognorm + self.safe_log(stage_3[len(element)][element]['suffixes'],stage_3[len(element)]['suffixes']),
                        "stems": s1_lognorm + s2_lognorm + self.safe_log((stage_3[len(element)][element]['stemix']/ ((stage_3[len(element)][element]['prefixes'] + stage_3[len(element)][element]['suffixes']) + 1)),stage_3[len(element)]['stemix'])
                    }
        #Final stage, gaussian distribution and sorting!
        allgrams2 = {'prefixes':[], 'suffixes':[], 'stems':[]}
        for v in transformed.values():
            for k2, v2 in v.items():
                for k3, v3 in v2.items():
                    if k3 == 'stems':
                        prob = self.gaussian(len(k2), v3, mu = 4.5, sigma=.75)
                        if len(k2) < 3: prob -= 10
                        allgrams2[k3].append((k2, prob))
                    else:
                        allgrams2[k3].append((k2, self.gaussian(len(k2), v3)))
        
        for k, v in allgrams2.items():
            transformed[k] = sorted(v, key= lambda x:x[1], reverse=True)
        if stems:
            with open(stems, 'r') as inj:
                inject_stems = json.load(inj)
            paired_stems = []
            for word in inject_stems:
                if str(len(word)) in transformed:
                    if word in transformed[str(len(word))]:
                        transformed[str(len(word))][word]['stems'] = 0 #i.e. MOST possible
                        paired_stems.append((word,0))
            print(f"updated the probs of {len(paired_stems)} stems")
        else:
            paired_stems = []
        """with open("nep_dict_1.json", "w") as outj:
            json.dump([word for word, prob in paired_stems], outj, indent=4, ensure_ascii=False)"""
        return transformed, paired_stems
    
    def gaussian(self, n, x, mu = 2, sigma = 1.5):
        gaussian_multiplier = np.exp(-((n - mu)**2) / (2 * sigma**2))
        # Since `x` is already in log-scale, add the log of the Gaussian multiplier to `x`
        adjusted_probability = x + np.log(gaussian_multiplier)
        return adjusted_probability
    
    def side_hustle(self, tokens, stems, entities, counts):
        with open(stems, "r") as inj:
            stems = json.load(inj)
        if type(stems) == dict:
            return stems
        if entities:
            with open(entities, 'r') as inj:
                entities = json.load(inj)
        #stems = [s[0] for s in stems]
                # Initialize the dictionary with a key for non-fitting tokens
        stem_dict = {'non-fitting': []}

        # Sort the stems by length in descending order to check the longest first
        stems_sorted = sorted(stems, key=len, reverse=True)

        # Iterate over each token
        for token in tqdm(tokens, desc="Finding largest stem per token"):
            # Initialize to keep track if a stem fits
            found_fit = False

            # Check each stem to see if it fits in the token
            for stem in stems_sorted:
                if stem in token:
                    # If the stem fits and it's the longest so far, add the token under this stem's key
                    if stem not in stem_dict:
                        stem_dict[stem] = []
                    stem_dict[stem].append(token)
                    found_fit = True
                    # Since we want the longest, no need to check shorter stems
                    break
            
            # If no stem fits, add the token to the non-fitting list
            if not found_fit:
                if entities:
                    if token in entities:
                        print(f"filtering out an entity: {token}")
                        continue
                    elif counts[token] > .0007:
                        print(f"filtering out a stopword: {token}")
                        continue
                    else:
                        stem_dict['non-fitting'].append(token)

        with open("side_hustle.json", "w") as outj:
            json.dump(stem_dict, outj, indent=4, ensure_ascii=False)
        print(len(stem_dict['non-fitting']))
        return stem_dict



if __name__ == "__main__":
    mytiztikz = Tiztikz('data/source_texts/nepali_bible_reformatted.txt', grammar = 'new_best_grammar.json', allgrams= 'allgrams.json', stems = 'data/side_hustle.json', entities = 'data/extracted_entities.json') #, ,  allgrams='allgrams.json'