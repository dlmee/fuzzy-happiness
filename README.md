# fuzzy-happiness

### How to Process

First ensure that you have the right format for tokenization. 

Then run through Tiztikz. You can have some extra bells and whistles, the most important being you can pass a dictionary. See format.

After you have a grammar built that you're happy with you are going to want to process it. 

* utility script check grammar. There are some options here. You may want to do afxremoval. You will need to collect the affixes by looking at the statistical affixes required to build the new best grammar.
* You will then want to pass this to the LLM to validate, define and do stopwords. You'll need to do two runs, one with all the instances that are 2 or more mappings per stem, and then a batched version for all the 1:1 mappings (i.e. tiztikz couldn't provide a decomposition)
* You'll then want to run format_llm_response.py, I've made it so that it can take two jsons, and it will return a single dictionary with all stems and LLM provided definitions. There is also merge.py, but I'm pretty sure format_llm_response.py has better logic...clean up and merge (ha!) if necessary. 
* This is the final_swahili.json (or LWC). You'll want to produce an inverse of this, for which there is a script, and then you're going to want to do a mapping, which will require the Bible again. Almost there!