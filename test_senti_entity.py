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

@app.route('/entities',methods=['POST'])
def entity():
    with open('entity_model.pkl', 'rb') as model_file:
        nlp = pickle.load(model_file)

    data=request.json

    text=data['text']
    
    doc = nlp(text)

    news_entites = []

    for ent in doc.ents:
        if ent.label_=="ORG":
            news_entites.append(ent.text)

    return jsonify(news_entites)

@app.route('/senti',methods=['GET'])
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

    with open('news-model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    def predict(sentence):
        # We need Token IDs and Attention Mask for inference on the new sentence
        test_ids = []
        test_attention_mask = []

        # Apply the tokenizer
        encoding = preprocessing(sentence, tokenizer)

        # Extract IDs and Attention Mask
        test_ids.append(encoding['input_ids'])
        test_attention_mask.append(encoding['attention_mask'])
        test_ids = torch.cat(test_ids, dim=0)
        test_attention_mask = torch.cat(test_attention_mask, dim=0)

        # Forward pass, calculate logit predictions
        with torch.no_grad():
            output = model(test_ids.to(device), token_type_ids=None, attention_mask=test_attention_mask.to(device))
        logits = output.logits.cpu().numpy().flatten()
        
        # Apply softmax to convert logits to probabilities
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

    # Join all sentences into a single string
    content = ' '.join(text_sentences)
    print(content)

    #content = "To ensure seamless travel experience and avoid inconvenience at toll plazas, National Highways Authority of India (NHAI) has advised Paytm FASTag users to procure a new FASTag issued by another bank before March 15, 2024. This will help in avoiding penalties or any double fee charges while commuting on National Highways, said Ministry of Road Transport & Highways on March 13.In line with the guidelines issued by the Reserve Bank of India (RBI) regarding restrictions on Paytm Payments Bank, the Paytm FASTags users will not be able to recharge or top-up the balance post March 15, 2024. However, they can use their existing balance to pay toll beyond the stipulated date.For any further queries or assistance related to Paytm FASTag, users can reach out to their respective banks or refer to the FAQs provided on the Indian Highways Management Company Limited (IHMCL) website, said MoRTH. NHAI has urged all Paytm FASTag users to take proactive measures to ensure a seamless travel experience on the National Highways across the country.Last month, the RBI had advised customers as well as merchants of Paytm Payments Bank Ltd (PPBL) to shift their accounts to other banks by March 15.REA"
    summary = summarize(content)

    # Print the summary
    print(summary)
    #new_sentence = "A bomb crashed on Syria"
    sentiment, score = predict(summary)

    score = float(score)

    return jsonify({'sentiment':sentiment,'score':score})
if __name__ == '__main__':
    app.run(debug=True,port=4001)