import streamlit as st
# Documentation: https://docs.streamlit.io/ 

import pickle
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# object of PorterStemmer
ps = PorterStemmer()

def text_transform(text):
    '''
    Preprocesses the input text and transforms it to be inserted into the model
    '''
    text = text.lower()
    text = nltk.word_tokenize(text)

    temp_list = []
    for i in text:
        if i.isalnum():
            temp_list.append(i)

    text = temp_list[:]
    # dont do this text=temp_list cause thats cloning and clear will remove values in text as well
    temp_list.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            temp_list.append(i)

    text = temp_list[:]
    temp_list.clear()

    for i in text:
        temp_list.append(ps.stem(i))
    
    return " ".join(temp_list)

# main func
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier...")

input_message = st.text_area("Please enter your email/sms here.")

if st.button('Predict'):
    # steps: 
    # 1. preprocess
    transformed_message = text_transform(input_message)

    # 2. vectorize
    vector_message = tfidf.transform([transformed_message])

    # 3. predict
    result = model.predict(vector_message)[0]

    # 4. display
    if result == 1:
        st.header("SPAM!")
    else:
        st.header("NOT SPAM!")


# run in terminal:
# to execute the file too: python app.py
# 1. to download streamlit: pip install streamlit
# 2. to download nltk: pip install nltk
# python -m nltk.downloader punkt
# streamlit run app.py (<nameOfFile>.py)