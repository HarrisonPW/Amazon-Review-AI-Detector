from flask import Flask, request, jsonify
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
import torch
import pickle
from flask_cors import CORS
import requests
import google.generativeai as genai
import joblib
import re
import nltk
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load GPT-2 model and tokenizer
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
gpt2_model = GPT2ForSequenceClassification.from_pretrained('openai-community/gpt2', num_labels=2)
gpt2_model.load_state_dict(torch.load('gpt2_spam_detector.pth'))
gpt2_model.to(device)

# Load LogisticRegression model and CountVectorizer
with open('logistic_regression_logistic_spam_detector.pkl', 'rb') as f:
    nb_model = pickle.load(f)

with open('logistic_regression_count_vectorizer.pkl', 'rb') as f:
    nb_vectorizer = pickle.load(f)

# load multinomialNB model and predict the spam by input review text
def predict_multinomialNB(review):
    cv = joblib.load('training_models/MultinomialNaiveBayes/cv.pkl')
    mnb = joblib.load('training_models/MultinomialNaiveBayes/MultinomialNB.pkl')
    ps = PorterStemmer() # initializing porter stemmer

    corpus=[]
    sentences=[]
    review=re.sub('[^a-zA-Z]',' ', review)
    review=review.lower()
    list=review.split()
    review=[ps.stem(word) for word in list if word not in set(stopwords.words('english'))]
    sentences=' '.join(review)
    corpus.append(sentences)

    x = cv.transform(corpus).toarray()

    result = mnb.predict(x)

    return "Spam" if result[0] == 1 else "Not Spam"



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


# LogisticRegression Model
def predict_spam_lr(review, model, vectorizer):
    review_vectorized = vectorizer.transform([review])
    prediction = model.predict(review_vectorized)
    return "Spam" if prediction[0] == 1 else "Not Spam"

# Gemini Model API
def predict_with_gemini_api(review):
    model_id = "tunedModels/reviewclassifier-g8uk4no67udl"
    genai.configure(api_key="AIzaSyBnhosxPFjzV6Vthnr9krUEJPqe8K5-cHo")

    try:
        model = genai.GenerativeModel(model_id)
        response = model.generate_content(review, safety_settings=[
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        }
    ])
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
    result_lr = predict_spam_lr(review, nb_model, nb_vectorizer)
    result_gemini = predict_with_gemini_api(review)
    result_nb3 = predict_spam_lr(review, nb_model, nb_vectorizer)  # TODO
    result_nb4 = predict_spam_lr(review, nb_model, nb_vectorizer)  # TODO

    # zilu added:
    result_mnb = predict_multinomialNB(review)

    # Calculate the percentage of spam predictions
    spam_predictions = [result_gpt2, result_lr, result_gemini, result_nb3, result_nb4]
    spam_count = sum([1 for prediction in spam_predictions if prediction == "Spam"])
    spam_percentage = (spam_count / len(spam_predictions)) * 100

    return jsonify({
        "review": review,
        "result_gpt2": result_gpt2,
        "result_lr": result_lr,
        "result_gemini": result_gemini,
        "result_nb3": result_nb3,
        "result_nb4": result_nb4,
        "spam_percentage": spam_percentage,
        "multinomialNB": result_mnb
    })


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
