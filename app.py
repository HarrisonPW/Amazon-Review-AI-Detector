import os
import pickle
import re
import json
from datetime import datetime

import google.generativeai as genai
import joblib
import requests
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify
from flask_cors import CORS
from confluent_kafka import Producer
from nltk.stem.porter import PorterStemmer
from safetensors.torch import load_file
from transformers import AutoTokenizer, BertModel
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification

app = Flask(__name__)
CORS(app)


class BertLSTMClassifier(torch.nn.Module):
    def __init__(self, model_name, num_labels=2, hidden_size=768, lstm_hidden_size=256, num_lstm_layers=1):
        super(BertLSTMClassifier, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained(model_name)
        self.lstm = torch.nn.LSTM(input_size=hidden_size, hidden_size=lstm_hidden_size, num_layers=num_lstm_layers,
                                  batch_first=True, bidirectional=False)
        self.classifier = torch.nn.Linear(lstm_hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        lstm_output, _ = self.lstm(sequence_output)
        lstm_output = lstm_output[:, -1, :]
        logits = self.classifier(lstm_output)
        return logits


model_dir = './'
tokenizer = AutoTokenizer.from_pretrained(model_dir)
BERTmodel = BertLSTMClassifier(model_name='bert-base-uncased', num_labels=2)
BERTmodel_state_dict = load_file(f'{model_dir}BERTLSTM.safetensors')
BERTmodel.load_state_dict(BERTmodel_state_dict)
BERTmodel.eval()

BERTmodel_state_dict = load_file(f'{model_dir}BERTLSTM.safetensors')
BERTmodel.load_state_dict(BERTmodel_state_dict)
BERTmodel.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BERTmodel.to(device)


# BERT prediction
def predict_with_BERTLSTM(text):
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=256,
        padding='max_length',
        return_tensors='pt',
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        logits = BERTmodel(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    return "Spam" if predicted_class == 1 else "Not Spam", probabilities.cpu().numpy()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load GPT-2 model and tokenizer
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
gpt2_model = GPT2ForSequenceClassification.from_pretrained('openai-community/gpt2', num_labels=2)
gpt2_model.load_state_dict(torch.load('gpt2_spam_detector.pth'))
gpt2_model.to(device)

# Load LogisticRegression model and CountVectorizer
with open('logistic_regression_logistic_spam_detector.pkl', 'rb') as f:
    lr_model = pickle.load(f)

with open('logistic_regression_count_vectorizer.pkl', 'rb') as f:
    lr_vectorizer = pickle.load(f)


# load multinomialNB model and predict the spam by input review text
def predict_multinomialNB(review):
    cv = joblib.load('training_models/MultinomialNaiveBayes/cv.pkl')
    mnb = joblib.load('training_models/MultinomialNaiveBayes/MultinomialNB.pkl')
    ps = PorterStemmer()

    corpus = []
    sentences = []
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    list = review.split()
    review = [ps.stem(word) for word in list]
    sentences = ' '.join(review)
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



# Kafka Configuration
conf = {
    'bootstrap.servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:29092'),
    'client.id': 'spam_detection_producer',
    'enable.idempotence': True,
    'acks': 'all',
    'retries': 5,
    'max.in.flight.requests.per.connection': 1
}

producer = Producer(conf)


def delivery_report(err, msg):
    """Callback for message delivery"""
    if err is not None:
        print(f'Message delivery failed: {err}')
    else:
        print(f'Message delivered to {msg.topic()} [{msg.partition()}]')


def create_topic():
    from confluent_kafka.admin import AdminClient, NewTopic
    admin_client = AdminClient({'bootstrap.servers': 'localhost:9092'})
    topic = NewTopic(
        "spam_detection_topic",
        num_partitions=1,
        replication_factor=1
    )
    try:
        admin_client.create_topics([topic])
        print("Topic created successfully")
    except Exception as e:
        print(f"Topic creation failed or already exists: {e}")


def send_to_kafka(review_text, spam_percentage):
    message = {
        'review': review_text,
        'is_spam': 1 if spam_percentage > 50 else 0,
        'spam_percentage': spam_percentage,
        'timestamp': datetime.utcnow().isoformat()
    }
    message_bytes = json.dumps(message).encode('utf-8')
    producer.produce('spam_detection_topic',
                     value=message_bytes,
                     callback=delivery_report)
    producer.poll(0)
    producer.flush(10)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'review' not in data:
        return jsonify({"error": "No review provided"}), 400

    review = data['review']
    result_gpt2 = predict_spam_gpt2(review, gpt2_model, gpt2_tokenizer, device)
    result_lr = predict_spam_lr(review, lr_model, lr_vectorizer)
    result_gemini = predict_with_gemini_api(review)
    result_mnb = predict_multinomialNB(review)
    result_BERTLSTM, probabilities = predict_with_BERTLSTM(review)

    # Calculate the percentage of spam predictions
    spam_predictions = [result_gpt2, result_lr, result_gemini, result_mnb, result_BERTLSTM]
    spam_count = sum([1 for prediction in spam_predictions if prediction == "Spam"])
    spam_percentage = (spam_count / len(spam_predictions)) * 100

    try:
        send_to_kafka(review, spam_percentage)
    except Exception as e:
        print(f"Error sending to Kafka: {e}")
        return jsonify({"error": "Error processing review"}), 500

    return jsonify({
        "review": review,
        "result_gpt2": result_gpt2,
        "result_lr": result_lr,
        "result_gemini": result_gemini,
        "result_mnb": result_mnb,
        "result_BERTLSTM": result_BERTLSTM,
        "spam_percentage": spam_percentage,
        "is_spam": 1 if spam_percentage > 50 else 0
    })


if __name__ == "__main__":
    try:
        create_topic()
        test_message = json.dumps({'test': 'connection'}).encode('utf-8')
        producer.produce('spam_detection_topic',
                         value=test_message,
                         callback=delivery_report)
        producer.flush()
        print("Successfully connected to Kafka")

        app.run(host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"Error starting the application: {e}")
    finally:
        producer.flush()
