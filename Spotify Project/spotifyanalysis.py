#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 14:51:29 2021

@author: ethanmitten
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv('/Users/ethanmitten/Desktop/Data Analytics/Python Projects/Spotify Project/train_spotifyclean.csv')
test =  pd.read_csv('/Users/ethanmitten/Desktop/Data Analytics/Python Projects/Spotify Project/test_spotifyclean.csv')

train.info()
#No missing values. We will go ahead and drop the song id column from the train and test dataset as it is not important to us anymore
#(No merging needs to be done)

train.head()
#danceability is listed twice so looking at the head it can be confirmed that these columns seem to be similar which
#means that the danceability.1 column will need to be dropped

train.time_signature.value_counts()
#From the value counts it can be seen that it is going to be unnecessary to include this column as it will just 
#cause more noise for the model since all, but five entries have a 4 time signature

f, ax = plt.subplots(figsize=(14,14))
sns.heatmap(train.corr(), linewidth=.5)

#Length and accousticness seem to be like the ones that are most closely related to liked songs
#Energy and loudness seem to be very similar

sns.countplot(x="release_date", data=train)
#Although not to informative because of how many dates there are in the dataset. There are a couple release_dates
#that have 5 songs and one that has 6 songs


def boxplot_maker(yvar):
    sns.boxplot(x="liked", y=yvar , data=train)
#Making a formulas to view the data in boxplot form easier

boxplot_maker('danceability')
#Seems like the more danceability there is to a song the more likely I am to dislike that song, but the margin between the two are so small that it would be hard to base
#my music selections solely off of a song's dancebility

boxplot_maker('popularity')
#Popularity seems to have almost no weight on whether I would prefer a song or not. There is a little more spread
#for the liked songs. I have a higher range of liking a song if the song is not as popular as opposed to if it is

boxplot_maker('length')
#With the length of a song it looks like overall the lengths are pretty similar. The whiskers on the boxplot for liked
#songs extend out a little further in the direction of longer songs, but for the most part a lot of similarity

boxplot_maker('acousticness')
#One of the variables that was seen on the matrix correlation plot as having a little more of an explanation of 
#liked songs was acousticness. However, there is not a huge difference in the middle parts of liked and disliked songs.
#Again where the biggest difference comes into play is that the extension of the whiskers is longer for my liked songs

boxplot_maker('energy')
#The energy variable is really not going to be of help to the model and may be a column to consider dropping when 
#trying to make a better model. The medians are almost exact and the whiskers are almost the same length. Plus 
#from the correlation plot it is already known that the energy variable is highly correlated to loudness anyway

boxplot_maker('instrumentalness')
#Instrumentalness looks to have quite a few outliers. It might not be a great column for explaining likes either

boxplot_maker('liveness')
#Liveness is looking pretty similar with disliked songs have a little more range in liveness

boxplot_maker('loudness')
#Loudness looks like the preferred column compared to energy. There seems to be more variation in this column
#than a lot of the other columns that have been looked at so far. The median is still pretty close together, but the 
#IQR is bigger for liked songs as well as the lower whisker of the liked songs

boxplot_maker('speechiness')
#Only real noticeable difference here is that disliked songs possess slightly higher median and a bigger IQR

boxplot_maker('tempo')
#Tempo has the biggest difference in median that has been seen out of all the other variables. It looks like
#I have preferred slower tempo songs over higher tempo songs. Althout the whiskers and IQR are similar the median 
#is a really interesting part of this boxplot


#Considerations Arising From Analysis

#1. The columns song_id, danceability.1, and time_signature should be dropped for sure as they add no value
#   into what is being looked for.

#2. Other columns that could be dropped to improve the usefulness of this model are the energy column,
#   instrumentalness and speechiness