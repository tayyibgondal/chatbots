from torch.utils.data import Dataset
import json


class ChatDataset(Dataset):
    def __init__(self, path: str, tokenizer):
        # Open file and read the data.
        self.data = json.load(open(path, "r"))

        # Filter texts from the data
        self.X = []
        for dialog in self.data:
            for sentence_info in dialog['dialog']:
                self.X.append(sentence_info['text'])

        # Changing the text format for inputting it to GPT model.
        for idx, text in enumerate(self.X):
            try:
                self.X[idx] = "<startofstring> " + text + \
                    " <bot>: " + self.X[idx+1] + " <endofstring>"
            except:
                break

        self.X = self.X[:5000]

        self.X_encoded = tokenizer(
            self.X, max_length=40, truncation=True, padding="max_length", return_tensors="pt")
        self.input_ids = self.X_encoded['input_ids']
        self.attention_mask = self.X_encoded['attention_mask']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attention_mask[idx])
