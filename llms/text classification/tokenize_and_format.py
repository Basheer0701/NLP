
from datasets import load_dataset
from transformers import BertTokenizer
import torch

# 1. Load AG News dataset
dataset = load_dataset("ag_news")

# 2. Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 3. Tokenize text samples
def tokenize_function(example):
    return tokenizer(
        example["text"],
        padding="max_length",       # pad to max length
        truncation=True,            # truncate long samples
        max_length=128              # limit input size to 128 tokens
    )

# 4. Apply tokenization to entire dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 5. Set the format for PyTorch tensors
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# 6. Check a sample
print(tokenized_dataset["train"][0])


