#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 01:36:26 2021

@author: creativechemicals
"""

import pandas as pd
import numpy as np

train_data = pd.read_csv("fake-news/train.csv")
test_data = pd.read_csv("fake-news/test.csv")

# Treating Null value
train_data.isna().sum() 

train_data = train_data.dropna()

#Splitting train data into x and y

y = train_data[['label']].reset_index(drop=True)
x = train_data.drop('label', axis=1).reset_index(drop=True)

#Processing data

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer


ps = PorterStemmer()

corpus = []
for sent in x['title']:
    sent = re.sub(r"[^a-zA-Z]"," ",sent)
    sent = word_tokenize(sent)
    sent = [ps.stem(word.lower()) for word in sent if not word in set(stopwords.words('english'))]
    sent = " ".join(sent)
    corpus.append(sent)

cv = CountVectorizer(max_features=5000,ngram_range=(1,3))
corpus = cv.fit_transform(corpus).toarray()
print(cv.get_feature_names()[:20])
#Splitting data

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(corpus, y, test_size=0.2)
        
# Model

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score

modelNB = MultinomialNB()
modelNB.fit(x_train, y_train)

y_pred = modelNB.predict(x_test)

print(f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

from sklearn.linear_model import PassiveAggressiveClassifier
linear_clf = PassiveAggressiveClassifier(max_iter=1000)
linear_clf.fit(x_train, y_train)

y_pred = linear_clf.predict(x_test)

print(f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

from sklearn.linear_model import RidgeClassifier
modelR = RidgeClassifier()
modelR.fit(x_train, y_train)

y_pred = modelR.predict(x_test)

print(f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

from sklearn.linear_model import SGDClassifier
modelSGD = SGDClassifier()
modelSGD.fit(x_train, y_train)

y_pred = modelSGD.predict(x_test)

print(f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

      
        

   
        
        
        
        
        
        