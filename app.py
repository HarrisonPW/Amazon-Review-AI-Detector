from flask import Flask, request, jsonify
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
import torch
import pickle
from flask_cors import CORS
import requests
import google.generativeai as genai


app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load GPT-2 model and tokenizer
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
gpt2_model = GPT2ForSequenceClassification.from_pretrained('openai-community/gpt2', num_labels=2)
gpt2_model.load_state_dict(torch.load('gpt2_spam_detector.pth'))
gpt2_model.to(device)

# Load MultinomialNB model and CountVectorizer
with open('multinomialNB_nb_spam_detector.pkl', 'rb') as f:
    nb_model = pickle.load(f)

with open('multinomialNB_count_vectorizer.pkl', 'rb') as f:
    nb_vectorizer = pickle.load(f)


def predict_spam_gpt2(review, model, tokenizer, device, max_length=128):
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


# MultinomialNB Model
def predict_spam_nb(review, model, vectorizer):
    review_vectorized = vectorizer.transform([review])
    prediction = model.predict(review_vectorized)
    return "Spam" if prediction[0] == 1 else "Not Spam"

# Gemini Model API
def predict_with_gemini_api(review):
    model_id = "tunedModels/reviewclassifier-g8uk4no67udl"  
    genai.configure(api_key="AIzaSyBnhosxPFjzV6Vthnr9krUEJPqe8K5-cHo")

    try:
        model = genai.GenerativeModel(model_id)
        response = model.generate_content(review) 
        prediction = response.text
        return "Spam" if prediction == 1 else "Not Spam"
    except requests.RequestException as e:
        print(f"Error calling Gemini API: {e}")
        return "Error"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'review' not in data:
        return jsonify({"error": "No review provided"}), 400

    review = data['review']
    result_gpt2 = predict_spam_gpt2(review, gpt2_model, gpt2_tokenizer, device)
    result_nb = predict_spam_nb(review, nb_model, nb_vectorizer)
    result_gemini = predict_with_gemini_api(review)
    result_nb3 = predict_spam_nb(review, nb_model, nb_vectorizer)  # TODO
    result_nb4 = predict_spam_nb(review, nb_model, nb_vectorizer)  # TODO

    # Calculate the percentage of spam predictions
    spam_predictions = [result_gpt2, result_nb, result_gemini, result_nb3, result_nb4]
    spam_count = sum([1 for prediction in spam_predictions if prediction == "Spam"])
    spam_percentage = (spam_count / len(spam_predictions)) * 100

    return jsonify({
        "review": review,
        "result_gpt2": result_gpt2,
        "result_nb": result_nb,
        "result_gemini": result_gemini,
        "result_nb3": result_nb3,
        "result_nb4": result_nb4,
        "spam_percentage": spam_percentage
    })


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
