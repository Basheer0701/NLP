from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F

print("🔍 Loading model and tokenizer...")

# ✅ Path to your trained model checkpoint
model_path = "results/checkpoint-250"  # adjust if needed

# ✅ Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

print("✅ Model loaded.")

# ✅ Sample input texts
texts = [
    "The stock market crashed due to global economic slowdown.",
    "Lionel Messi scored a hat trick in the final match.",
    "A new species of dinosaur was discovered in Argentina.",
    "Microsoft announced a new version of Windows."
]

# ✅ AG News labels
label_names = ["World", "Sports", "Business", "Sci/Tech"]

print("🔍 Starting inference...")

# ✅ Predict and print results
for text in texts:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1)
        predicted = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted].item()
        print(f"\n📰 Input: {text}")
        print(f"📌 Prediction: {label_names[predicted]} ({confidence:.2%})")

print("\n✅ Inference complete.")
