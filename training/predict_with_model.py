from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# inputs = tokenizer(["Hello, my dog is cute", "hi, my dog is cuter", "hi, my cat is the"], return_tensors="pt")
# labels = torch.tensor([1, 2, 3]).unsqueeze(0)  # Batch size 1
# outputs = model(**inputs, labels=labels)

print(model.predict("hello, my dog is cute"))

# print(labels)

# print(outputs)