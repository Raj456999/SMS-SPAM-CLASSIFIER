from html.parser import piclose
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import streamlit as st
import pickle
import string
nltk.download('stopwords')
ps=PorterStemmer()

tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))
print('lucky draw' in tfidf.get_feature_names_out())
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
    # res=model.predict(vector_input)[0]
    probability=model.predict_proba(vector_input)[0]
    spam_probability=probability[1]
    ham_probability=probability[0]
    prediction='Spam' if spam_probability>ham_probability else 'Not Spam'
    confidence=max(spam_probability,ham_probability)*100
    st.progress(int(confidence))
    st.write(f"Prediction :{prediction}")
    st.write(f"Confidence :{confidence:.2f}%")
    st.write(spam_probability,ham_probability)
    st.write('New model loaded')

if st.button('Predict'):
    tranformed_sms=text_tranform(sms_input)

    #Vectorization

    vector_input=tfidf.transform([tranformed_sms])

    #Prediction by model
    # res=model.predict(vector_input)[0]
    probability=model.predict_proba(vector_input)[0]
    spam_probability=probability[1]
    ham_probability=probability[0]
    prediction='Spam' if spam_probability>ham_probability else 'Not Spam'
    confidence=max(spam_probability,ham_probability)*100
    st.progress(int(confidence))
    st.write(f"Prediction :{prediction}")
    st.write(f"Confidence :{confidence:.2f}%")




    
    # Displaying results
    # if res==1:
    #     st.header('Spam')
    # else:
    #     st.header('Not Spam')
