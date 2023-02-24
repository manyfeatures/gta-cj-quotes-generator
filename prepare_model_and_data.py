from torch.utils.data import Dataset  # this is the pytorch class import
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from infer import *

# s1 = 'WAKA WAKA QB WTF BBBQ WAKA LOREM IPSUM WAKA'.split()
# s2 = 'WAKA OMFG QB WTF WAKA WAKA LOREM IPSUM WAKA'.split()
# print(s1)
# print(s2)
# print(nltk.edit_distance(s1, s2))
#
def save_model(tokenizer, model):
    # save model checkpoint
    tokenizer.save_pretrained("model")
    model.save_pretrained("model")


def prepare_dataset(tokenizer, txt_list):
    train_ds = CustomDataset(txt_list, tokenizer, max_length=15) # max len 15 is empirical
    print('Dataset length: {}'.format(len(train_ds)))
    train_dataloader = DataLoader(
        train_ds,
        sampler = RandomSampler(train_ds), # Sampling for training is random
        batch_size = 16
    )
    return train_dataloader


class CustomDataset(Dataset):
    def __init__(self, txt_list, tokenizer, max_length):
        self.tokenizer = tokenizer  # the gpt2 tokenizer we instantiated
        self.input_ids = []
        self.attn_masks = []

        for txt in txt_list:
            encodings_dict = tokenizer('CJ quote:' + txt + '<|endoftext|>',
                                       truncation=True,
                                       max_length=max_length,
                                       padding="max_length",
                                       return_tensors='pt')
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]


def load_model(model_path):
    """it loads model"""
    model_name = 'distilgpt2'
    print(model_name)

    if not os.path.exists(model_path):
        print("model has not been loaded yet")
        model_path = model_name

    model = GPT2LMHeadModel.from_pretrained(model_path)#.to(DEVICE)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path,
                                              bos_token='CJ quote:',
                                              eos_token='<|endoftext|>', # default?
                                              pad_token='<|pad|>')
    model.resize_token_embeddings(len(tokenizer))  # so as to add new words, tokens?
    save_model(tokenizer, model) # for saving locally?

    return tokenizer, model
