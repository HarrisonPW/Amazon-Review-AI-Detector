from flask import Flask, request, jsonify
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
import torch

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = GPT2ForSequenceClassification.from_pretrained('openai-community/gpt2', num_labels=2)
model.load_state_dict(torch.load('gpt2_spam_detector.pth'))
model.to(device)


def predict_spam(review, model, tokenizer, device, max_length=128):
    model.eval()
    encoding = tokenizer.encode_plus(
        review,
        add_special_tokens=True,
        max_length=max_length,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        prediction = torch.argmax(outputs.logits, dim=1)

    return "Spam" if prediction.item() == 1 else "Not Spam"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'review' not in data:
        return jsonify({"error": "No review provided"}), 400

    review = data['review']
    result = predict_spam(review, model, tokenizer, device)
    return jsonify({"review": review, "prediction": result})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
