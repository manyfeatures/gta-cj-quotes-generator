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


### Inference

```
I'mma spend this on a good meal.
You better apologize before I get out and hit me, bitch!
You're out your mind!
Come on, lady out of the hospital!
You hit me, punk-ass!
Yeah, you dropped a bomb on me! (You Dropped a bomb on me! (You Dro
Now, you better shut up!
I don't want y'all doing nothing funny to me at the time, too.
Oh, you got car-slipping again!
I'm having fun this for now?
You got a life to do, mister?
Yeah, you think your hard?
I can't walk now, sucker!
Yeah, what's up now, I'm just a fat slob, a fat slob!
Get outta here punk! Now!
Get lost, now, or I'll take you out, punk-ass!
Come on, you need to exercise more exercise.
I don't give a shit, fool!
I don't look like a bitch you don't need this no more.
Come on, lady outta there!
```

quite similar to original dataset, sometime those sentences are the exact copies
