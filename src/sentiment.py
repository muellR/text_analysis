from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def analyse_sentiment(text: str) -> dict:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = F.softmax(outputs.logits, dim=-1)
    sterne = torch.argmax(probs).item() + 1  # 1-5 Sterne
    konfidenz = probs[0][sterne - 1].item()
    
    return {
        "text": text,
        "sterne": sterne,
        "konfidenz": f"{konfidenz:.2%}",
        "label": "positiv" if sterne >= 4 else "negativ" if sterne <= 2 else "neutral"
    }

# Test
print(analyse_sentiment("Super schnelle Lieferung, bin aber nicht sehr zufrieden!"))
# {'text': '...', 'sterne': 5, 'konfidenz': '85.3%', 'label': 'positiv'}