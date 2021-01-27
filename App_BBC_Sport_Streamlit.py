import numpy as np
import pickle
import os
import pandas as pd
import streamlit as st 
from PIL import Image
import re
import string
# stop words
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
import textblob
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

#=========================

# load transform TF_IDF
vectorizer_in = open("Feature_engineering_methods/vectorizer.pkl","rb")
vectorizer=pickle.load(vectorizer_in)

# load transform TF_IDF Optimise
vectorizer_in = open("Feature_engineering_methods/vectorizer_Optimise.pkl","rb")
vectorizer_Optimise=pickle.load(vectorizer_in)

# load transform countVectors
countVectors_in = open("Feature_engineering_methods/countVectors.pkl","rb")
countVectors=pickle.load(countVectors_in)

#@app.route('/')
def welcome():
    return "Welcome All"

#==========================
# Netoyage et Lemmatizer avec Tokenisation
def nettoyer_text(df):
    df["text_nettoyer"] = df["text"].map(lambda x: x.lower())
    # Remplacement de la ponctuation basée sur des opérations d'expressions régulières
    df["text_nettoyer"] = df['text_nettoyer'].map(lambda x: re.sub(r'\'|\\n\\n| n |\\n|\\|-',' ',x))
    # Remplacer deux éspaces par un seul éspace
    df["text_nettoyer"] = df['text_nettoyer'].map(lambda x: x.replace('  ', ' '))
    # Supprimer la ponctuation basé sur string.punctuation (string.punctuation == !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~)
    df["text_nettoyer"] = df["text_nettoyer"].map(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    #Supprimer les stopwords
    mots_stop = stopwords.words('english')
    df["text_nettoyer"] = df["text_nettoyer"].map(lambda x: " ".join([mot for mot in x.split() if not mot in mots_stop])) 
    #Lemmatizer avec Tokenisation
    lemmatizer = WordNetLemmatizer()
    df['text_tok_lim'] = df['text_nettoyer'].map(lambda x: 
                                                 " ".join(lemmatizer.lemmatize(mot) for mot in TextBlob(x).words))
    return df
#===========================

#@app.route('/Term Frequency (TF-IDF)',methods=["Get"])
def transform_TF_IDF(data, vectorizer):
    vectors = vectorizer.transform(data)
    feature_names = vectorizer.get_feature_names()

    dense = vectors.todense()
    denselist = dense.tolist()

    X_texte_Tf_Idf = pd.DataFrame(denselist, columns=feature_names)
    
    return X_texte_Tf_Idf
#===========================

#@app.route('/Word Count Vectors',methods=["Get"])
def transform_countVectors(data):
    vectors = countVectors.transform(data)
    feature_names = countVectors.get_feature_names()

    dense = vectors.todense()
    denselist = dense.tolist()

    X_texte_countVectors = pd.DataFrame(denselist)
    
    return X_texte_countVectors
#===========================

#@app.route('/predict',methods=["Get"])
def predict_TF_IDF(X_texte_Tf_Idf, classifier):
    # load ML models
    pickle_in = open("ML_model/"+classifier+"_TF_IDF.pkl","rb")
    model=pickle.load(pickle_in)
    
    prediction=model.predict(X_texte_Tf_Idf)
    return prediction

#===========================

#@app.route('/predict',methods=["Get"])
def predictX_countVectors(X_texte_countVectors, classifier):
    # load ML models
    pickle_in = open("ML_model/"+classifier+"_countVectors.pkl","rb")
    model=pickle.load(pickle_in)
    
    prediction=model.predict(X_texte_countVectors)
    return prediction

#===========================
def main():
    #background
    page_bg_img = '''
    <style>
    body {
    background-image: url("https://hicenter.co.il/wp-content/uploads/2016/01/bkg.jpg");
    background-size: cover;
    }
    </style>
    '''
    #st.markdown(page_bg_img, unsafe_allow_html=True)

    st.write("""
    ## **Classification of articles based on the topical area** 
    ## ------------- --------------------------------------------
    This App predicts the **likelihood** that a document belongs to a certain  topical area.

    """)
    st.write("""
    To train the machine learning model, we used the **BBCSport** dataset. The data file contains 737 documents from the BBC Sport website corresponding to sports news articles in five topical areas.

Class Labels: 5 (Athletics, Cricket, Football, Rugby, Tennis)
    """)
    """
    @author: **Bilal Khomri**
    """
    
    html_temp = """
    <div style="background-color:#ff9966;padding:10px">
    <h2 style="color:white;text-align:center;">Text Classification Form</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    
    # App
    st.write("""### **Sport Text**""")
    sport_text = st.text_input('Please ! Write Your Text Here ', '')
    #txt = st.text_area(label, value='', height=500, max_chars=None, key=None)
    df= pd.DataFrame({"text":[sport_text]})
    
    st.write("""### **Approaches to transforming text into a feature space**""")
    transf = st.radio("",('TF–IDF Vectors', 'Word Count Vectors'))
    
    st.write("""### **Machine learning models**""")
    classifier = st.radio("",('GradientBoostingClassifier', 'Xgboost','RandomForestClassifier','Naive_bayes','AdaBoostClassifier'))

    html_temp = """
    <div style="background-color:#ff9966;padding:5px;margin-bottom:20px"> </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    result=""
    if st.button("Predict"):
        if len(sport_text) < 3:
            st.warning('Please ! Fill in the text below !')
        else:
            nettoyer_text(df)
            if transf == 'TF–IDF Vectors':
                if classifier == 'GradientBoostingClassifier':
                    X_texte_Tf_Idf = transform_TF_IDF(df['text_tok_lim'], vectorizer_Optimise)
                else:
                    X_texte_Tf_Idf = transform_TF_IDF(df['text_tok_lim'], vectorizer)
                
                result= predict_TF_IDF(X_texte_Tf_Idf, classifier)
                if result[0] == 'athletics' :
                    image = Image.open('img/athletics.jpg')
                    st.image(image, caption='Article about Athletics', use_column_width=False, width = 500)
                if result[0] == 'football' :
                    image = Image.open('img/football.jpg')
                    st.image(image, caption='Article about Football', use_column_width=False, width = 500)
                if result[0] == 'rugby' :
                    image = Image.open('img/rugby.jpg')
                    st.image(image, caption='Article about Rugby', use_column_width=False, width = 500)
                if result[0] == 'cricket' :
                    image = Image.open('img/cricket.jpg')
                    st.image(image, caption='Article about Cricket', use_column_width=False, width = 500)     
                if result[0] == 'tennis' :
                    image = Image.open('img/tennis.jpg')
                    st.image(image, caption='Article about Tennis', use_column_width=False, width = 500)

            if transf == 'Word Count Vectors':
                X_texte_countVectors = transform_countVectors(df['text_tok_lim'])
                result= predictX_countVectors(X_texte_countVectors, classifier)
                if result[0] == 'athletics' :
                    image = Image.open('img/athletics.jpg')
                    st.image(image, caption='Article about Athletics', use_column_width=False, width = 500)
                if result[0] == 'football' :
                    image = Image.open('img/football.jpg')
                    st.image(image, caption='Article about Football', use_column_width=False, width = 500)
                if result[0] == 'rugby' :
                    image = Image.open('img/rugby.jpg')
                    st.image(image, caption='Article about Rugby', use_column_width=False, width = 500)
                if result[0] == 'cricket' :
                    image = Image.open('img/cricket.jpg')
                    st.image(image, caption='Article about Cricket', use_column_width=False, width = 500)     
                if result[0] == 'tennis' :
                    image = Image.open('img/tennis.jpg')
                    st.image(image, caption='Article about Tennis', use_column_width=False, width = 500)
            
if __name__=='__main__':
    main()
