from transformers import AdamW
from infer import generate_eval_sample
from prepare_model_and_data import save_model
from tqdm import tqdm

# TODO
def eval():
    pass

def print_sample_output(step, batch_loss, model, tokenizer):
    print(f'Batch {step}, Batch train loss:{batch_loss}')
    generate_eval_sample(model, tokenizer)


def train(model, tokenizer, train_dataloader, val_dataloader, device, epochs=20):
    training_stats = []
    optimizer = AdamW(model.parameters(),
                      lr=5e-4,
                      eps=1e-8
                      )
    sample_every_batch = 20
    #total_steps = len(train_dataloader) * epochs
    for epoch_i in range(0, epochs):

        print(f'Epoch {epoch_i + 1}/{epochs}')

        total_train_loss = 0
        model.train()  # mode
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            assert model.training, 'print model is not in training mode'
            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_masks = batch[1].to(device)

            outputs = model(b_input_ids,
                            labels=b_labels,
                            attention_mask=b_masks,
                            token_type_ids=None
                            )

            loss = outputs[0]
            batch_loss = loss.item()
            total_train_loss += batch_loss
            # Get sample every 100 batches.
            if step % sample_every_batch == 0 and not step == 0:
                print_sample_output(step, batch_loss, model, tokenizer)
                model.train()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            #model.zero_grad()
        # model.eval()
        #
        # total_eval_loss = 0
        # #nb_eval_steps = 0
        # # Evaluate data for one epoch
        # for batch in val_dataloader:
        #     b_input_ids = batch[0].to(device)
        #     b_labels = batch[0].to(device)
        #     b_masks = batch[1].to(device)
        #     with torch.no_grad():
        #         outputs = model(b_input_ids,
        #                         attention_mask=b_masks,
        #                         labels=b_labels)
        #         loss = outputs[0]
        #     batch_loss = loss.item()
        #     total_eval_loss += batch_loss
        # avg_val_loss = total_eval_loss / len(val_dataloader)
        # print(f'Validation loss: {avg_val_loss}')
        #
        # # Calculate the average loss over all of the batches.
        # avg_train_loss = total_train_loss / len(train_dataloader)
        #training_stats.append(avg_train_loss)
        save_model(tokenizer, model)
