from html.parser import piclose
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import streamlit as st
import pickle
import string

ps=PorterStemmer()

tfidf=pickle.load(open('C:\\Users\\palla\\OneDrive\
\Desktop\\Projects\\First\\SMS Spam Classifier\\vectorizer.pkl','rb'))
model=pickle.load(open('C:\\Users\\palla\\OneDrive\\Desktop\\Projects\\First\\SMS Spam Classifier\\model.pkl','rb'))
st.title('SMS Spam Classifier')
sms_input=st.text_area('Enter the Message here')

#Preprocessing the data
import string
def text_tranform(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    res=[]
    for i in text:
        if i.isalnum():
            res.append(i)
    text=res[:]
    res.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            res.append(i)
    text=res[:]
    res.clear()
    for i in text:
        res.append(ps.stem(i))
    return ' '.join(res)
if st.button('Predict'):
    tranformed_sms=text_tranform(sms_input)

    #Vectorization

    vector_input=tfidf.transform([tranformed_sms])

    #Prediction by model

    res=model.predict(vector_input)[0]

    #Displaying results

    if res==1:
        st.header('Spam')
    else:
        st.header('Not Spam')