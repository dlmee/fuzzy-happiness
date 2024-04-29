Dillon Mee
Ling 696 Presentation

### Yet Another look at Unsupervised Morphological Analysis

A topic that has long interested me is unsupervised morphological analysis, which is perhaps a folly because in many contexts (especially high resource contexts) this can be seen as a solved problem. It is perhaps doubly so a folly because morphology by its very definition is interested in units of meaning, and the meaning itself is very likely opaque to any kind of statistical analysis.

Consider some of the pragmatic linguistic discussions we've had in class. One can, for example, suggest that a 'lemma' or dictionary form be defined by utility. Such a definition could be made, and that definition would be not so much right or wrong, as useful or not-useful. The same cannot be said for morphology. There are morphological parses which are just flat-out wrong. Let's take the extreme example. One possible parse for morphology would be:

> m-o-r-p-h-o-l-o-g-y

This is of course, wrong, and we know it's wrong because `m` by itself doesn't mean anything. We know that a better parse is: 

> morph-ology 

We know this because we know that morph means something--something like shape--not because we see morph all over the place in the language. 

But as Goldsmith, 2001 notes: "Developing an unsupervised learner using raw text data as its sole input offers several attractive aspects, both theoretical and practical. At its most theoretical, un-
supervised learning constitutes a (partial) linguistic theory, producing a completely explicit relationship between data and analysis of that data." That's a compelling thought!  

### What's already been done (by others(20 years ago))

Goldsmith, 2001 utilizes Minimum Description Length (MDL), especially around signatures, which can be thought of as morphological patterns that apply to the same stems. For example one signature might be [NULL.ed.ing.s] in which each of the four morphemes present may combine with words such accent, add, or alert. These signatures in combination with Minimum Description Length and some heuristics produce a fairly powerful unsupervised morphological parser. 

Schone & Jurafsky, 2000 use latent semantic analysis to explore a morphology. Basically a morph + affix is only accepted if they are semantically similar by creating vectors to be analyzed for distance akin to distributional semantics. 

Creutz and Lagus, 2002 also use MDL, but with some interesting additions, claiming to be especially suitable for agglutinative languages. Similar to Goldsmith the aim is to describe the data with a concise 'codebook'. Notably they use a 'dreaming' stage which "Due to the online learning, as the number of processed words increases, the quality of the set of morphs in the codebook gradually improves. Consequently, words encountered in the beginning of the input data, and not observed since, may have a sub-optimal segmentation in the new model, since at some point more suitable morphs have emerged in the codebook." I think this idea is fascinating, and provokes the question: to what extent does our model need to be able to re-evaluate itself?

Additionally I appreciated their discussion on what segments verses morphs, "The practical purpose of the segmentation is to provide a vocabulary of language units that is smaller and generalizes better than a vocabulary consisting of words as they appear in text. Such a
vocabulary could be utilized in statistical language modeling, e.g., for speech recognition. Moreover, one could assume that such a discovered morph vocabulary would correspond rather closely to linguistic morphemes of the language." 

### My approach


#### Bibliography

Schone, P., & Jurafsky, D. (2000). Knowledge-free induction of morphology using latent semantic analysis. In Fourth Conference on Computational Natural Language Learning and the Second Learning Language in Logic Workshop.

