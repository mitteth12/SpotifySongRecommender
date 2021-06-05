#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 17:34:56 2021

@author: ethanmitten
"""

import pandas as pd

train = pd.read_csv('/Users/ethanmitten/Desktop/Data Analytics/Python Projects/Spotify Project/train_spotifyclean.csv')
test =  pd.read_csv('/Users/ethanmitten/Desktop/Data Analytics/Python Projects/Spotify Project/test_spotifyclean.csv')

train_original = train.copy()
test_original = test.copy()

y = train.liked.values

train.drop(['liked', 'song_id', 'release_date', 'danceability.1', 'time_signature'], axis=1, inplace=True)



from sklearn.model_selection import train_test_split
X_train, X_cv, y_train, y_cv = train_test_split(train, y, test_size =.3)


#Logistic Regression
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
LogisticRegression()

from sklearn.metrics import accuracy_score
log_pred_cv = log_model.predict(X_cv)
accuracy_score(y_cv, log_pred_cv)
#57% accuracy on basic logistic attempt


train.drop(['energy', 'instrumentalness', 'speechiness'], axis=1, inplace=True)

X_train1, X_cv1, y_train1, y_cv1 = train_test_split(train, y, test_size = .3)
log_model.fit(X_train1, y_train1)
log_pred1_cv = log_model.predict(X_cv1)
accuracy_score(y_cv1, log_pred1_cv)
#53% accuracy when dropping the columns that were closely related

#Set Back Model Analysis
#
#Things to think about for creating the next version of this model:
    #
    #The similarities between the songs I dislike and like seem to be causing problems in increasing the 
    #accuracy. This could indicate a couple different things. One it could indicate simply I statistically 
    #do not have much of a difference between the likes and dislikes. In that case there is not much that 
    #can be done because I simply have almost randomness in the possibilites. 
    
#Things that can be done moving forward:
    #
    #The first one that jumps out immediately is increasing the sample size. I based my conclusions off of fifty disliked songs
    #and above 80 liked songs. If the model is able to have a better spread than the accuracy could be increased.
    #
    #The second possibility is just gathering more data. This could mean doing encoding with artists and release_years. This could
    #also mean gathering more data from other sources besides Spotify to collect information on the specific songs being looked at.
    
