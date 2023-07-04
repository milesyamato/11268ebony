import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR

# Initialize two pretrained models
model_name1 = "TheBloke/guanaco-13B-SuperHOT-8K-fp16"
model_name2 = "TheBloke/Wizard-Vicuna-13B-Uncensored-SuperHOT-8K-fp16"

tokenizer1 = AutoTokenizer.from_pretrained(model_name1)
tokenizer2 = AutoTokenizer.from_pretrained(model_name2)

model1 = AutoModel.from_pretrained(model_name1)
model2 = AutoModel.from_pretrained(model_name2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StackedModel(nn.Module):
    def __init__(self, model1, model2):
        super(StackedModel, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.classifier = nn.Linear(5120, 2)  # Adjust dimensions as needed

    def forward(self, input_ids, attention_mask):
        outputs1 = self.model1(input_ids=input_ids, attention_mask=attention_mask)
        inputs2 = outputs1.last_hidden_state
        outputs2 = self.model2(inputs=inputs2)
        output = self.classifier(outputs2.last_hidden_state[:, 0, :])  # Apply a classifier on the [CLS] token
        return output

stacked_model = StackedModel(model1, model2).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(stacked_model.parameters())
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  # Learning rate scheduler

class MyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        return input_ids, attention_mask, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.texts)

# Load your data
df = pd.read_csv(os.environ["DATASET_PATH"])
texts = df[os.environ["TEXT_COLUMN"]].tolist()
labels = df[os.environ["LABEL_COLUMN"]].tolist()

train_texts, val_test_texts, train_labels, val_test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
val_texts, test_texts, val_labels, test_labels = train_test_split(val_test_texts, val_test_labels, test_size=0.5, random_state=42)

train_dataset = MyDataset(train_texts, train_labels, tokenizer1)
val_dataset = MyDataset(val_texts, val_labels, tokenizer1)
test_dataset = MyDataset(test_texts, test_labels, tokenizer1)

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 64))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

def calculate_accuracy(preds, labels):
    _, predictions = torch.max(preds, 1)
    correct = (predictions == labels).float()
    acc = correct.sum() / len(correct)
    return acc

num_epochs = int(os.environ.get("NUM_EPOCHS", 10))
best_val_loss = float('inf')
early_stop_counter = 0

for epoch in range(num_epochs):
    stacked_model.train()
    for input_ids, attention_mask, labels in train_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        outputs = stacked_model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    stacked_model.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in val_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = stacked_model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            acc = calculate_accuracy(outputs, labels)

            val_loss += loss.item()
            val_acc += acc.item()

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(stacked_model.state_dict(), os.environ["MODEL_SAVE_PATH"])
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= 5:
            break

stacked_model.load_state_dict(torch.load(os.environ["MODEL_SAVE_PATH"]))

test_loss = 0
test_acc = 0
with torch.no_grad():
    for input_ids, attention_mask, labels in test_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        outputs = stacked_model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        acc = calculate_accuracy(outputs, labels)

        test_loss += loss.item()
        test_acc += acc.item()

print(f'Test Loss: {test_loss/len(test_loader):.3f}, Test Acc: {test_acc/len(test_loader):.3f}')



