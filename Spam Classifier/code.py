#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 01:49:45 2021

@author: creativechemicals
"""

import pandas as pd
import numpy as np

data = pd.read_csv("spam.txt", sep='\t', names=['label','message'])
print(data.head(10))

print(data['label'].value_counts())

for i in range(len(data)):
    data.label[i]=1 if data.label[i]=='spam' else 0
    
# Text Preprocessing

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import re

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def processing(data):
    processed_data = data[['label','message']]
    for i in range(len(data)):
        processed_data.message[i] = re.sub(r"[^a-zA-Z0-9]"," ",processed_data.message[i])
        #str1 = ''
        str_list = word_tokenize(processed_data.message[i])
        str_list = [ps.stem(j) for j in str_list if not j in stop_words]
        str_list = ' '.join(str_list)
        processed_data.message[i]=str_list
        
    return processed_data
    
processed_data = processing(data)
print(processed_data.head(10))


y = processed_data['label'].astype('int') 
x = processed_data.message

# BagOfWords

from sklearn.feature_extraction.text import CountVectorizer 

cv = CountVectorizer()
train_x_cv = cv.fit_transform(x).toarray()


# TF-IDF

from sklearn.feature_extraction.text import TfidfVectorizer


tf_idf = TfidfVectorizer()
train_x_tfidf = tf_idf.fit_transform(x).toarray()

# Model 

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

def RFmodel(x, y):
    
    X_train, X_test, y_train, y_test= train_test_split(x,y,test_size=0.2,random_state=1)
    model = RandomForestClassifier(n_estimators=200, criterion='entropy')

    model.fit(X_train, y_train)
    y_pred=model.predict(X_test)




    print("Classification Report")
    print(classification_report(y_test, y_pred))
    print("\nConfussion Matrix")
    print(confusion_matrix(y_test, y_pred))
    print("\nAccuracy")
    print(accuracy_score(y_test, y_pred))
    print("\nf1 score")
    print(f1_score(y_test, y_pred, average='macro'))
    
def NBmodel(x, y):
    
    X_train, X_test, y_train, y_test= train_test_split(x,y,test_size=0.2,random_state=1)
    model2 = MultinomialNB().fit(X_train, y_train)

    y_pred=model2.predict(X_test)




    print("Classification Report")
    print(classification_report(y_test, y_pred))
    print("\nConfussion Matrix")
    print(confusion_matrix(y_test, y_pred))
    print("\nAccuracy")
    print(accuracy_score(y_test, y_pred))
    print("\nf1 score")
    print(f1_score(y_test, y_pred, average='macro'))
    

print("\nRandom Forest Model with Count vectorized data")
RFmodel(train_x_cv, y)


print("\nRandom Forest Model with Tf-idf data")
RFmodel(train_x_tfidf, y)

print("\nNaive Bayes Model with Count vectorized data")
NBmodel(train_x_cv, y)


print("\nNaive Bayes Model with Tf-idf data")
NBmodel(train_x_tfidf, y)















