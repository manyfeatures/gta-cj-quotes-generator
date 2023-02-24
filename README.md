# gta-cj-quotes-generator

### TODO List

1) Add metrics
   - ROUGE?
   - BLEU?
   - perplexity
   - other
   - plot it and save graphs
   - word count
   - number of old sentence in generated outputs
   - how to use it for random strategies?
   - find the closest sentence in the dataset for the given output, count all cases
   - k-nn for sentence from the output
   - use classifier to filter CJ phrases
   - plot it, and save to file
   - add Leveinstein distance from nltk
   - word cloud 
   
2) top_k in manual inference can be replaced with any other method for diagnostics  
3) Augmentation
4) Data cleaning ?
5) Different architectures
6) Which texts datasets have structure to check generation correctness?
   - Maybe rhymes, code, etc
7) Dialogue generation, how to implement?
   - Style Transfer for making CJ converstional bot: https://medium.com/nlplanet/two-minutes-nlp-quick-intro-to-text-style-transfer-61de9cbd4083
8) Emotions classificaton for gta dialogue, key words in sentence
9) Try different techniques for generation
10) Visualize attention to show important words
11) add argparse
