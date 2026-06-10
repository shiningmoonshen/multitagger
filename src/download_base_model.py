# only to be run once, to download the base model and save it locally for offline training. After running this, you can run the training script without an internet connection.
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "distilbert-base-uncased"
LOCAL_DIR = "models/base/distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=10)

tokenizer.save_pretrained(LOCAL_DIR)
model.save_pretrained(LOCAL_DIR)
print(f"Saved to {LOCAL_DIR} — training can now run fully offline.")