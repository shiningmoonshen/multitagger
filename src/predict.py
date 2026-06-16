import json
import re
import random
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

FINAL_DIR = Path("models/distilbert-v1/final")
LABEL_MAP_PATH = Path("data/processed/label_map.json")
MAX_LENGTH = 128
CONFIDENCE_THRESHOLD = 0.65
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

_model = None
_tokenizer = None
_id2label: dict[int, str] | None = None
_device = None

_REDACTION = re.compile(r"X{2,}")
_WHITESPACE = re.compile(r"\s+")


def _clean(text: str) -> str:
    text = _REDACTION.sub("", text)
    text = text.lower()
    text = text.strip()
    text = _WHITESPACE.sub(" ", text)
    return text


def load_model() -> None:
    global _model, _tokenizer, _id2label, _device
    if _model is not None:
        return

    with open(LABEL_MAP_PATH, encoding="utf-8") as f:
        label_map: dict[str, int] = json.load(f)
    _id2label = {idx: name for name, idx in label_map.items()}

    if torch.cuda.is_available():
        _device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        _device = torch.device("mps")
    else:
        _device = torch.device("cpu")

    _tokenizer = AutoTokenizer.from_pretrained(str(FINAL_DIR), local_files_only=True)
    _model = AutoModelForSequenceClassification.from_pretrained(
        str(FINAL_DIR), local_files_only=True
    )
    _model.to(_device)
    _model.eval()


def predict_batch(texts: list[str]) -> list[dict]:
    if not texts:
        raise ValueError("texts must be a non-empty list")
    for i, t in enumerate(texts):
        if t is None or (isinstance(t, str) and not t.strip()):
            raise ValueError(f"texts[{i}] is null or empty — provide a non-empty narrative")

    load_model()

    cleaned = [_clean(t) for t in texts]

    enc = _tokenizer(
        cleaned,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    enc = {k: v.to(_device) for k, v in enc.items()}

    with torch.no_grad():
        logits = _model(**enc).logits
    probs = torch.softmax(logits, dim=-1)

    top2 = torch.topk(probs, k=2, dim=-1)

    results = []
    for i in range(len(texts)):
        top_prob = top2.values[i][0].item()
        results.append(
            {
                "tag": _id2label[top2.indices[i][0].item()],
                "confidence": round(top_prob, 4),
                "low_confidence": top_prob < CONFIDENCE_THRESHOLD,
                "top_2": [
                    {
                        "tag": _id2label[top2.indices[i][j].item()],
                        "confidence": round(top2.values[i][j].item(), 4),
                    }
                    for j in range(2)
                ],
            }
        )
    return results


def predict(text: str) -> dict:
    if text is None or (isinstance(text, str) and not text.strip()):
        raise ValueError("text is null or empty — provide a non-empty narrative")
    return predict_batch([text])[0]


if __name__ == "__main__":
    import json as _json

    examples = [
        (
            "Credit Reporting",
            "I have an inaccurate collection account on my credit report that does not belong "
            "to me. I have disputed this with the bureau multiple times but the error remains. "
            "This is severely impacting my ability to obtain a mortgage.",
        ),
        (
            "Mortgage",
            "My mortgage servicer applied my payment to the wrong month and is now claiming I "
            "am 30 days late. They have reported this delinquency to the credit bureaus and "
            "refuse to correct it despite proof of timely payment.",
        ),
        (
            "Debt Collection",
            "A debt collector has been calling me multiple times a day including before 8am "
            "and after 9pm. I sent a written cease-and-desist letter two weeks ago and they "
            "are still contacting me.",
        ),
        (
            "Bank Account",
            "My bank processed an unauthorized ACH debit from my checking account. I did not "
            "authorize this transaction and the bank is refusing to reverse the charge or "
            "initiate a dispute on my behalf.",
        ),
        (
            "Credit Card",
            "I was charged twice for the same purchase on my credit card. The merchant "
            "acknowledged the duplicate charge but my card issuer still has not issued a "
            "credit after 45 days.",
        ),
    ]

    print(f"Loading model from {FINAL_DIR} …\n")
    load_model()

    for expected_category, narrative in examples:
        result = predict(narrative)
        print(f"Expected category : {expected_category}")
        print(f"Narrative excerpt : {narrative[:80]}…")
        print(_json.dumps(result, indent=2))
        print()
