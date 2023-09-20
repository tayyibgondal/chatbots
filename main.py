from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm
import torch
from dataset import ChatDataset


def train(dataset, model, optim):

    epochs = 12

    for i in tqdm.tqdm(range(epochs)):
        for X, a in dataset:
            X = X.to(device)
            a = a.to(device)
            # Zero out the accumulated gradients
            optim.zero_grad()
            # Compute loss
            loss = model(X, attention_mask=a, labels=X).loss
            # Compute gradients
            loss.backward()
            # Apply gradients
            optim.step()
        torch.save(model.state_dict(), "model_state.pt")
        # Testing the quality of the model during training
        print(infer("Hello how are you?"))


def infer(inp):
    inp = "<startofstring> " + inp + " <bot>: "
    inp = tokenizer(inp, return_tensors="pt")
    X = inp["input_ids"].to(device)
    a = inp["attention_mask"].to(device)
    output = model.generate(X, attention_mask=a)
    output = tokenizer.decode(output[0])
    return output


device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({"pad_token": "<pad>",
                              "bos_token": "<startofstring>",
                              "eos_token": "<endofstring>"})
tokenizer.add_tokens(["<bot>:"])

model = GPT2LMHeadModel.from_pretrained("gpt2")
# Since we added new tokens to the tokenizer.
model.resize_token_embeddings(len(tokenizer))
# Transfer to cuda
model = model.to(device)

# print(tokenizer.decode(model.generate(**tokenizer("hey i was good at basketball but ",
#                          return_tensors="pt"))[0]))

dataset = ChatDataset("chatdata.json", tokenizer)
dataset = DataLoader(dataset, batch_size=64)

model.train()
optim = Adam(model.parameters(), lr=1e-3)
print("Training ....")
train(dataset, model, optim)

print("Infer from model : ")
while True:
    inp = input()
    print(infer(inp))
