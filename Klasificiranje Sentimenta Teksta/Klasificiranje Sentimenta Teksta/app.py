import streamlit as st
import pandas as pd
import pickle as pk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from nltk.corpus import stopwords


with open('models.pkl', 'rb') as f:
    vect, lr_model, nb_model, svc_model = pk.load(f)


def clean_review(review):
    return ' '.join(word for word in review.split() if word.lower() not in stopwords.words('english'))


def predict_sentiment(model, review):
    review = clean_review(review)
    review_vect = vect.transform([review])
    prediction = model.predict(review_vect)
    return prediction[0]

# Streamlit 
st.title('Analiza sentimenta teksta')

review = st.text_input('Upiši recenziju:')
model_option = st.selectbox('Choose a model:', ('Logistička regresija', 'Naive Bayes', 'Potporni vektor SVC'))

if st.button('Predict'):
    if model_option == 'Logistička regresija':
        prediction = predict_sentiment(lr_model, review)
    elif model_option == 'Naive Bayes':
        prediction = predict_sentiment(nb_model, review)
    elif model_option == 'Potporni vektor SVC':
        prediction = predict_sentiment(svc_model, review)
    st.write(f'Predikcija je: {prediction}')
