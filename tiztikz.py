import re
import json
import math
from tqdm import tqdm
import random
import copy
import itertools
from itertools import permutations

class Tiztikz:
    def __init__(self, corpus, allgrams = None):
        tokens = self.read_txt(corpus)
        self.allgrams = self.morph_merge(tokens, allgrams=allgrams)
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

    def morph_merge(self, tokens, allgrams=None):
        max_length = 0
        longest_word = ""
        for line in tokens:
            for word in line:
                if len(word) > max_length:
                    max_length = len(word)
                    longest_word = word
        print(f"the longest word is: {max_length}, {longest_word}")
        if allgrams:
            with open(allgrams, 'r') as inj:
                allgrams = json.load(inj)
        else:
            allgrams = self._first_pass(tokens)
            allgrams = self._second_pass(allgrams, tokens)
            with open("allgrams.json", 'w') as outj:
                json.dump(allgrams, outj, indent=4, ensure_ascii=False)
            
        allgrams = self._third_pass(allgrams, tokens)
        return allgrams

    def _first_pass(self, tokens):
        #Need to collect the types of words, to know what we have to be able to build. 
        allgrams = {'**word**': {'**total**': {}}}
        for line in tqdm(tokens, desc="First Pass"):
            for word in line:
                thisgrams = self._find_grams(word)
                for n, grams in thisgrams.items():
                    if n not in allgrams:
                        allgrams[n] = {"**total**": 0, '**threshold**': 0}
                    for gram in grams:
                        allgrams[n]["**total**"] += 1
                        if gram not in allgrams[n]:
                            allgrams[n][gram] = {'count': 1, 'prep': 0, 'sufp': 0, 'word':0, 'before': {'#':0}, 'after': {'#':0}, 'coverage': [word]}
                        else:
                            allgrams[n][gram]['count'] += 1
                            if word not in allgrams[n][gram]['coverage']:
                                allgrams[n][gram]['coverage'].append(word)
        return allgrams

    def _second_pass(self, allgrams, tokens):
        for line in tqdm(tokens, desc="Second pass"):
            for word in line:
                if not word: continue
                possibles = self.generate_splits_iterative(word)
                # = self.generate_splits_components(word)
                for stem, grams in possibles.items():
                    n = len(stem)
                    for key in ['before', 'after']:
                        for gram in list(grams[key]):
                            if gram not in allgrams[n][stem][key]:
                                allgrams[n][stem][key][gram] = 1
                            else:
                                allgrams[n][stem][key][gram] +=1

                    allgrams[n][stem]['prep'] = allgrams[n][stem]['before']['#'] / (sum([count for key, count in allgrams[n][stem]['before'].items() if key != '#']) +1)
                    allgrams[n][stem]['sufp'] = allgrams[n][stem]['after']['#'] / (sum([count for key, count in allgrams[n][stem]['after'].items() if key != '#']) + 1)
        return allgrams

    def _third_pass(self, allgrams, tokens):
        allgrams = self.transform_allgrams(allgrams)
        tokens = set(word for line in tokens for word in line)
        tokens = list(tokens)
        #Here is where I need to start building rules
        grammar = {
    'probability': float('-inf'),
    'cost': float('inf'),
    'stems': [],
    'prefixes':['NULL'],
    'suffixes':['NULL'],
    'combinations': []
}

        #let's initialized an unprobable and expensive grammar, every word and no affixes or stems!
        grammar = self.build_grammar(grammar, tokens, allgrams)
        efficiency = grammar['cost']
        improved = True
        t = 20
        counter = 0
        improved = True
        while t > 1:
            t -= 1
            grammars = self.ratchet_grammar(grammar, tokens, allgrams, temp=random.randint(1,t), growing=improved, paths=7)
            improved = False
            for k,v in grammars.items():
                print(f"The cost of {k} = {v['cost']}")
                if v['cost'] >= efficiency:
                    efficiency = v['cost']
                    grammar = v
                    improved = True
            print(f"Improved is {improved}")
        return grammar

                
    def build_grammar(self, grammar, tokens, allgrams):
        grammar['combinations'] = []
        for word in tqdm(tokens, desc="building grammar"):
            decomposition = self.recursive_decompose(word, grammar['prefixes'], grammar['stems'], grammar['suffixes'])
            if decomposition:
                grammar = self.shift(grammar, decomposition) #This makes used rules less likely to be dropped. 
                ppool = [p for p in decomposition['prefixes']]
                spool = [s for s in decomposition['suffixes']]
                if grammar['combinations']:
                    for prefix in decomposition['prefixes']:
                        for affixation in grammar['combinations']:
                            if affixation['prefix']:
                                if prefix == affixation['prefix']:
                                    affixation['stems'].append(decomposition['stem'])
                                    affixation['generates'].append((prefix + decomposition['stem'], word))
                                    ppool.remove(prefix)
                                    break
                    for suffix in decomposition['suffixes']:
                        for affixation in grammar['combinations']:
                            if affixation['suffix']:
                                if suffix in affixation['suffix']:
                                    affixation['stems'].append(decomposition['stem'])
                                    affixation['generates'].append((decomposition['stem'] + suffix, word))
                                    spool.remove(suffix)
                                    break
                for p in ppool:
                    #then we don't have a rule that matches this. We need to make a rule. 
                    grammar['combinations'].append({
                        'stems': [stem for stem in decomposition['stem'] if stem],
                        'prefix': p,
                        'suffix': None,
                        'probability': [0], #we'll need to build a custom function here :D 
                        'generates': [(p + decomposition['stem'], word)]
                    })
                for s in spool:
                    grammar['combinations'].append({
                        'stems': [stem for stem in decomposition['stem'] if stem],
                        'prefix': None,
                        'suffix': s,
                        'probability': [0], #we'll need to build a custom function here :D 
                        'generates': [(decomposition['stem'] + s, word)]
                    })
            else:
                grammar['combinations'].append(
                    {
            'stems': [word],
            'prefix': None,
            'suffix': None,
            'probability': [allgrams[str(len(word))][word]['stems']],
            'generates': [word]
        })
        grammar['probability'] = sum(p for c in grammar['combinations'] for p in c['probability'])
        grammar['cost'] = sum([len(v['generates']) for v in grammar['combinations']])/len(grammar['combinations'])
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
            found_components = {'prefixes': [], 'stem': [], 'suffixes': []}

        # Base case: If the target word matches any stem directly or no more target word to check
        if target_word in stems:
            if not found_components['prefixes'] and not found_components['suffixes']:
                return None
            found_components['stem'] = target_word #.append(target_word)
            #print(f"Final decomposition at depth {depth}: {found_components}")
            return found_components
        elif not target_word or depth > 10:  # Prevent infinite recursion
            #print(f"No decomposition found for '{target_word}' at depth {depth}")
            return None

        for prefix in prefixes:
            if target_word.startswith(prefix):
                # Update target_word by removing the prefix
                new_target = target_word[len(prefix):]
                found_components['prefixes'].append(prefix)

                # Recursive call
                return self.recursive_decompose(new_target, prefixes, stems, suffixes, found_components, depth + 1)

        for suffix in suffixes:
            if target_word.endswith(suffix):
                # Update target_word by removing the suffix
                new_target = target_word[:-len(suffix)]
                found_components['suffixes'].append(suffix)

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


    def ratchet_grammar(self, grammar, tokens, allgrams, temp=1, paths=3, growing=True):
        def mutate(mutation, allgrams, targets, temp):
            for k,v in targets.items():
                for i in range(v+1):
                    for combo in mutation['combinations']:
                        if len(combo[k]) > 2:
                            combinations = [(''.join(word), word) for word in list(itertools.product([el for el in combo[k] if len(el) < 4], repeat=2)) if ''.join(word) in allgrams[k]]
                            if combinations:
                                choice = random.choice(combinations)
                                for pfx in choice[1]:
                                    if pfx in mutation[k]:
                                        #print(f"removing {pfx}")
                                        mutation[k].remove(pfx)
                                    else:
                                        pass
                                        #print(f"already removed {pfx}")
                                mutation[k].append(choice[0])
                                break
                    if k in ['prefixes', 'suffixes']:
                        if 'NULL' not in mutation[k]:
                            mutation[k].append('NULL')
            return mutation
        
        def grow(additive, allgrams, targets, temp):
            for k,v in targets.items():
                for _ in range((v+1)*2):
                    additive[k].append(self.suggest_element(additive[k], allgrams[k], temp))
            return additive

        def shrink(purge, allgrams, targets, temp):
            for k,v in targets.items():
                for element in purge[k]:
                    count = sum([len(v['generates']) for v in purge['combinations'] if element in v[k]])
                    if count < 5:
                        purge[k].remove(element)
                    if k in ['prefixes', 'suffixes']:
                        if 'NULL' not in purge[k]:
                            purge[k].append('NULL')
            return purge
        
        pgrammars = {}
        for i in range(1, paths + 1):
            yes_no = [True, True, False]
            mypath = copy.deepcopy(grammar)
            if growing == True:
                targets = {'stems': 10*temp, 'prefixes':1*temp,'suffixes':1*temp}
                mypath = grow(mypath, allgrams, targets, temp)
            else:
                if random.choice(yes_no):
                    targets = {'stems': 2*temp, 'prefixes':1*temp,'suffixes':1*temp}
                    mypath = mutate(mypath, allgrams, targets, temp)
                if random.choice(yes_no):
                    targets = {'stems': 2*temp, 'prefixes':1*temp,'suffixes':1*temp}
                    mypath = shrink(mypath, allgrams, targets, temp)
            pgrammars[f"Possible {i}"] = self.build_grammar(mypath, tokens, allgrams)
        

        return pgrammars
    
    def suggest_element(self, previous, allgrams, temp):
        allgrams = list(allgrams.keys())
        confirmed = False
        counter = 0
        while not confirmed:
            counter += (20*temp)
            first = random.randint(0, counter)
            second = random.randint(0, counter)
            #third = random.randint(0,len(allgrams))
            if first - second >= 0:
                if first < len(allgrams):
                    if allgrams[first] not in previous:
                        confirmed = True
            if counter >= len(allgrams):
                return allgrams[0]
        return allgrams[first]



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


    def transform_allgrams(self, allgrams):
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
                    stage_3[len(k2)] = {'prefix':0, 'suffix':0, 'stemix':0}
                if k2 not in stage_3[len(k2)]:
                    stage_3[len(k2)][k2] = {'prefix':v2['before']['#'], 'suffix':v2['after']['#'], 'stemix':max([sum([p for k,p in v2['before'].items() if k != '#']) + sum([p for k, p in v2['after'].items() if k != '#']), min(v2['before']['#'], v2['after']['#'])])}
                else:
                    stage_3[len(k2)][k2]['prefix'] += v2['before']['#']
                    stage_3[len(k2)][k2]['suffix'] += v2['after']['#']
                    stage_3[len(k2)][k2]['stemix'] += max([(sum([p for k,p in v2['before'].items() if k != '#']) + sum([p for k, p in v2['after'].items() if k != '#'])), min(v2['before']['#'], v2['after']['#'])])
                stage_3[len(k2)]['prefix'] += v2['before']['#']
                stage_3[len(k2)]['suffix'] += v2['after']['#']
                stage_3[len(k2)]['stemix'] += max([(sum([p for k,p in v2['before'].items() if k != '#']) + sum([p for k, p in v2['after'].items() if k != '#'])), min(v2['before']['#'], v2['after']['#'])])
                print("And done!")
        print("And done!")
        
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
                        "prefix": s1_lognorm + s2_lognorm + self.safe_log(stage_3[len(element)][element]['prefix'],stage_3[len(element)]['prefix']),
                        "suffixes": s1_lognorm + s2_lognorm + self.safe_log(stage_3[len(element)][element]['suffix'],stage_3[len(element)]['suffix']),
                        "stems": s1_lognorm + s2_lognorm + self.safe_log(stage_3[len(element)][element]['stemix'],stage_3[len(element)]['stemix'])
                    }

        
        return transformed



if __name__ == "__main__":
    mytiztikz = Tiztikz('sg.txt', allgrams='allgrams.json') #, ,  allgrams='allgrams.json'
