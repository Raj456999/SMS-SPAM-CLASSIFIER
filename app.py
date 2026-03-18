from html.parser import piclose
import nltk
from networkx.algorithms.bipartite.generators import configuration_model
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import streamlit as st
import pickle
import string
nltk.download('stopwords')
ps=PorterStemmer()

tfidf=pickle.load(open('C:\\Users\\palla\\OneDrive\\Desktop\\Projects\\First\\SMS Spam Classifier\\vectorizer.pkl','rb'))
model=pickle.load(open('C:\\Users\\palla\\OneDrive\\Desktop\\Projects\\First\\SMS Spam Classifier\\model.pkl','rb'))
st.title('SMS Spam Classifier')
sms_input=st.text_area('Enter the Message here')

#Preprocessing the data
import string
# def text_tranform(text):
#     text=text.lower()
#     text=nltk.word_tokenize(text)
#     res=[]
#     for i in text:
#         if i.isalnum():
#             res.append(i)
#     text=res[:]
#     res.clear()
#     for i in text:
#         if i not in stopwords.words('english') and i not in string.punctuation:
#             res.append(i)
#     text=res[:]
#     res.clear()
#     for i in text:
#         res.append(ps.stem(i))
#     return ' '.join(res)
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ps = PorterStemmer()

def text_tranform(text):
    text = text.lower()
    
    # simple tokenization without nltk.word_tokenize
    words = text.split()
    
    # keep only alphanumeric words
    words = [w for w in words if w.isalnum()]
    
    # remove stopwords and punctuation
    words = [w for w in words if w not in stopwords.words('english') and w not in string.punctuation]
    
    # stemming
    # words = [ps.stem(w) for w in words]
    
    return ' '.join(words)

if st.button('Predict'):
    tranformed_sms=text_tranform(sms_input)
    #Vectorization
    vector_input=tfidf.transform([tranformed_sms])
    #Prediction by model
    prob=model.predict_proba(vector_input)[0]
    spam_prob=prob[1]
    ham_prob=prob[0]

    threshold=0.4

    prediction= 'Spam' if spam_prob>threshold else 'Not spam'
    if spam_prob>threshold:
        confidence=spam_prob*100
    else:
        confidence=ham_prob*100
    st.progress(int(confidence))
    # st.write(f"Prediction :{prediction}")
    st.write(f"confidence :{confidence:.2f}%")
    st.write(spam_prob,ham_prob)

    if prediction=='Spam':
        st.error(f"🚨 spam with confidence {confidence:.2f}%")
    else:
        st.error(f"😎 NOT spam with confidence {ham_prob*100:.2f}%")
