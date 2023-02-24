from train_dataset import *
import torch
import evaluate
import numpy as np

# from rouge_score import rouge_scorer, scoring
# test perplexity
# rouge = evaluate.load('rouge')
# blue = evaluate.load('bleu')

# taken from somewhere on the web
def topk(probs, n=30):
    """select 1 token from top k"""
    # The scores are initially softmaxed to convert to probabilities
    probs = torch.softmax(probs, dim=-1)
    # PyTorch has its own topk method, which we use here
    tokensProb, topIx = torch.topk(probs, k=n)
    # The new selection pool (10 choices) is normalized
    tokensProb = tokensProb / torch.sum(tokensProb)
    # Send to CPU for numpy handling
    tokensProb = tokensProb.cpu().detach().numpy()
    # Make a random choice from the pool based on the new prob distribution
    choice = np.random.choice(n, 1, p=tokensProb)
    tokenId = topIx[choice][0]

    return int(tokenId)

def model_infer(model, tokenizer, init_token="CJ quote:", max_length=20, device='cuda:0'):
    # Preprocess the init token (task designator)
    init_id = tokenizer.encode(init_token)
    result = init_id
    init_input = torch.tensor(init_id).unsqueeze(0).to(device)

    with torch.set_grad_enabled(False): # turn of the grad ?
        # Feed the init token to the model
        output = model(init_input)
        # Flatten the logits at the final time step
        logits = output.logits[0, -1]
        # Make a top-k choice and append to the result
        result.append(topk(logits))
        # For max_length times:
        for i in range(max_length):
            # Feed the current sequence to the model and make a choice
            input = torch.tensor(result).unsqueeze(0).to(device)
            output = model(input)
            logits = output.logits[0, -1]
            res_id = topk(logits)

            # If the chosen token is EOS, return the result
            if res_id == tokenizer.eos_token_id:
                return tokenizer.decode(result)
            else:  # Append to the sequence
                result.append(res_id)
    # IF no EOS is generated, return after the max_len
    return tokenizer.decode(result)

def inference(model, tokenizer):
    model.eval()
    results = set()
    while len(results) < 20:
        quote = model_infer(model, tokenizer).replace("CJ quote:", "").strip()
        if quote not in TEXTS.split('\n') and quote not in results:
            results.add(quote)
            CGREEN2 = '\33[92m'
            CEND = '\033[0m'
            print(CGREEN2 + quote + CEND)


def generate_eval_sample(model, tokenizer, txt=None, device='cuda:0'):
    '''default HF generation with input and without
       Works not good for now'''
    #print("Tokenizer parameters", tokenizer)
    print('SAMPLE OUTPUT\nsample input:', txt)

    model.eval()
    if not txt:
        print('default gen')
        # simple generator without output
        # TODO: resolve the problem with the same output
        sample_outputs = model.generate(bos_token_id=tokenizer.bos_token_id,
                                        pad_token_id=tokenizer.eos_token_id,
                                        do_sample=True, # use sampling from distribution and not argmax?
                                        max_length=20,
                                        top_k=50,
                                        top_p=0.95,
                                        num_beams=5,
                                        num_return_sequences = 10)
    else:
        encodings_dict = tokenizer('CJ quote:' + txt,
                                   truncation=True,
                                   max_length=20,  # empirical choice see below
                                   padding="max_length",
                                   return_tensors='pt',
                                   ).to(device)
        sample_outputs = model.generate(encodings_dict['input_ids'].to(device),
                                        attention_mask=encodings_dict['attention_mask'].to(device),
                                        pad_token_id=tokenizer.eos_token_id,
                                        do_sample=True,
                                        top_k=50,
                                        max_length=200,
                                        top_p=0.95,
                                        num_return_sequences = 10)

    print(sample_outputs.size())
    preds = tokenizer.batch_decode(sample_outputs, skip_special_tokens=True)
    # print(f'Example output: {preds}\n')
    # print('ROUGE METRICS:', rouge.compute(predictions=[preds], references=[txt]))
    # print('mean geometric precision for unigram 1-4')
    # what is length ratio?
    # print('BLEU METRICS:', blue.compute(predictions=[preds], references=[txt]))
    # print(''.join(['='*120]))

    CGREEN2 = '\33[92m'
    CEND = '\033[0m'
    print(CGREEN2 + "\n".join(preds) + CEND)
