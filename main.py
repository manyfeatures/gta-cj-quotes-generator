#!/home/solar/miniconda3/bin/python
# The GPT2 Model transformer with a language modeling head on top (linear layer with weights tied to the input embeddings).
from train import train
from prepare_model_and_data import *
from train_dataset import TEXTS
from sklearn.model_selection import train_test_split
#torch.manual_seed(42)


if __name__ == '__main__':
    # Model
    model_path = 'model'
    tokenizer, model = load_model(model_path)

    # Dataset
    txt_list = TEXTS.split('\n')
    train_text, val_text = train_test_split(txt_list, test_size=0.3, random_state=42)
    train_dataloader = prepare_dataset(tokenizer, train_text)
    val_dataloader = prepare_dataset(tokenizer, val_text) # we should process it in a different way

    device = 'cuda:0'
    model = model.to(device)

    print('Training:')
    train(model, tokenizer, train_dataloader, val_dataloader, device, epochs=20)

    print('Inference:')
    inference(model, tokenizer)
	
    # default methods from HF for generation
    # print('Generate quote:')
    # generate_eval_sample(model, tokenizer)
