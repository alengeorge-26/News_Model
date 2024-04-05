import flask
from flask import Flask,request,jsonify
from flask_restful import Resource,Api
from flask_cors import CORS
import spacy
from spacy import displacy
from spacy.tokens import DocBin
import pickle
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

app = Flask(__name__)
CORS(app)

api=Api(app)

with open('entity_model.pkl', 'rb') as model_file:
    nlp = pickle.load(model_file)

with open('news-model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/entities',methods=['POST'])
def entity():
    data=request.json

    text=data['text']
    
    doc = nlp(text)

    news_entites = []

    for ent in doc.ents:
        if ent.label_=="ORG":
            news_entites.append(ent.text)

    return jsonify(news_entites)

@app.route('/senti',methods=['POST'])
def senti():
    data=request.json
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS']= '0'


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading Tokenizer..... \nPlease wait...\n")
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased',
        do_lower_case=True
    )

    token_id = []
    attention_masks = []
    token_id = []
    attention_masks = []

    def preprocessing(input_text, tokenizer):
        return tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=32,
            padding=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

    def predict(sentence):
    
        test_ids = []
        test_attention_mask = []

        encoding = preprocessing(sentence, tokenizer)

        test_ids.append(encoding['input_ids'])
        test_attention_mask.append(encoding['attention_mask'])
        test_ids = torch.cat(test_ids, dim=0)
        test_attention_mask = torch.cat(test_attention_mask, dim=0)

        with torch.no_grad():
            output = model(test_ids.to(device), token_type_ids=None, attention_mask=test_attention_mask.to(device))
        logits = output.logits.cpu().numpy().flatten()
        
        probabilities = np.exp(logits) / np.sum(np.exp(logits))
        
        labels = np.argmax(logits)
        sentiment = ''
        if labels == 0:
            sentiment = 'Negative'
        elif labels == 1:
            sentiment = 'Neutral'
        elif labels == 2:
            sentiment = 'Positive'

        return sentiment, probabilities[labels]

    def summarize(text, max_sentences=3):
        sentences = text.split('.')
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(sentences)
        sentence_vectors = vectors.toarray()

        combined_text = '.'.join(sentences)
        combined_vector = vectorizer.transform([combined_text]).toarray()

        sentence_scores = []
        for sentence_vector in sentence_vectors:
            sentence_score = cosine_similarity(sentence_vector.reshape(1, -1), combined_vector)[0][0]
            sentence_scores.append(sentence_score)

        top_sentences = sorted(zip(sentence_scores, sentences), reverse=True)[:max_sentences]
        summary = '. '.join([sentence for _, sentence in top_sentences]) + '.'
        return summary

    text_sentences = data['text_sentences']

    content = ' '.join(text_sentences)

    summary = summarize(content)

    sentiment, score = predict(summary)

    score = float(score)

    return jsonify({'sentiment':sentiment,'score':score})

if __name__ == '__main__':
    app.run(debug=True,port=4001)