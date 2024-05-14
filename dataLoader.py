import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def load_jsonl(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def prepare_data_for_model(data):
    model_inputs = []
    labels = []
    for item in data:
        sentence1 = item['sentence1']
        sentence2 = item['sentence2']
        label = item['gold_label']
        model_inputs.append((sentence1, sentence2))
        labels.append(label)
    return model_inputs, labels


def split_data(model_inputs, labels, train_size, test_size):
    # Split data into training and remaining data
    inputs_train, inputs_remaining, labels_train, labels_remaining = train_test_split(
        model_inputs, labels, train_size=train_size, random_state=42)

    # Split the remaining data equally into validation and test sets
    test_val_size = 0.5  # Since remaining is 20% of data, half of it will be 10%
    inputs_val, inputs_test, labels_val, labels_test = train_test_split(
        inputs_remaining, labels_remaining, train_size=test_val_size, random_state=42)

    return inputs_train, labels_train, inputs_val, labels_val, inputs_test, labels_test

class NLIDataset(Dataset):
    def __init__(self, inputs, labels, tokenizer, max_length):

        # Filter out entries where 'gold_label' is '-'
        filtered_inputs = []
        filtered_labels = []
        for inp, lbl in zip(inputs, labels):
            if lbl != "-":
                filtered_inputs.append(inp)
                filtered_labels.append(lbl)

        self.inputs = filtered_inputs
        self.labels = filtered_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = f"Sentence1: {self.inputs[idx][0]} Sentence2: {self.inputs[idx][1]} Relationship:"
        label = self.label_map[self.labels[idx]]

        encoding = self.tokenizer.encode_plus(
            input_text, 
            add_special_tokens=True,
            max_length=self.max_length,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        return encoding.input_ids[0], torch.tensor(label, dtype=torch.long)