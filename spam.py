import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

ps = PorterStemmer()

# Load the TF-IDF vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Function to transform text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

st.title("SMS Spam Detection")

input_sms = st.text_input("Enter the Message")

if st.button('Predict'):

    transformed_sms = transform_text(input_sms)

    vector_input = tfidf.transform([transformed_sms])

    # Load the training data
    df = pd.read_csv('C:/Users/91779/Desktop/running/sms.csv')

    X_train = df['message']
    y_train = df['label ']

    # Fit the model with training data
    X_train = tfidf.transform(X_train)
    model.fit(X_train, y_train)

    # Reshape vector_input to (1, n_features)
    vector_input = vector_input.reshape(1, -1)

    print("Vector input shape:", vector_input.shape)
    print("Vector input:", vector_input)

    result = model.predict(vector_input)

    print("Predicted result:", result)

    if result[0] == "spam":
        st.header("Spam")
    else:
        st.header("Not Spam")
