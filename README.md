# TODO List

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
   - word cloud in progress
   
2) top_k in manual inference can be replaced with any other method for diagnostics  
2) ~~Don't split sentences in small pieces, unite it to make it longer~~
   - it doesn't look nice
3) Val dataset
   - [x] something has been added
   - [x] save model each epoch
4) Augmentation
5) Data cleaning ?
6) Different architectures
7) Which texts datasets have structure to check generation correctness?
   - Maybe rhymes, code, etc
8) Dialogue generation, how to implement?
   - Style Transfer for making CJ converstional bot: https://medium.com/nlplanet/two-minutes-nlp-quick-intro-to-text-style-transfer-61de9cbd4083
9) Emotions classificaton for gta dialogue, key words in sentence
10) Try different techniques for generation
11) Visualize attention to show important words
12) add argparse