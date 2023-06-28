#!/usr/bin/env python
# coding: utf-8

# In[13]:


# import libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


# In[3]:


# Reading ratings file from github
ratings = pd.read_csv('C:/Users/User/Desktop/ANINDITADAS_SentimentRecommendedBasedProject/sample30.csv' , encoding ='latin-1')
ratings.head()  


# In[36]:


# Dividing the dataset into train and test
from sklearn.model_selection import train_test_split
train, test = train_test_split(ratings, test_size=0.30, random_state=31)

ratings.info() 


# In[6]:


print(train.shape) 


# In[7]:


print(test.shape)    


# In[16]:


import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OneHotEncoder
from pandas.core.reshape.pivot import pivot


# In[20]:



#pivot = df.piv# pivot the train eatings dataset into matrix format in which columns are categories and the rows are ids 
df_pivot = train.pivot_table(index='id', columns='categories', values= 'reviews_rating').fillna(0)
 
#cols = pivot.columns.to_flat_index().str.join('_')

df_pivot.head(4)  


# In[21]:


# creating dummy train & dummy test datasets 

# copy the train dataset into dummy dummy_train
dummy_train = train.copy()

dummy_train.head() 


# In[23]:


# The categories not rated by user is marked as 1 for prediction
dummy_train['reviews_rating'] = dummy_train['reviews_rating'].apply(lambda x: 0 if x>=1 else 1) 


# In[35]:


# convert the dummy train dataset into matrix format
dummy_train = train.pivot_table(index='id', columns='categories', values= 'reviews_rating').fillna(1)     


# In[63]:


dummy_train.head()      


# In[37]:


# User similarity Matrix

# Using cosine similarity
df_pivot.index.nunique() 


# In[38]:


from sklearn.metrics.pairwise import pairwise_distances 


# In[40]:


# creating the user similarity Matrix using pairwise_distance function
user_correlation = 1 - pairwise_distances(df_pivot, metric= 'cosine')
user_correlation[np.isnan(user_correlation)] = 0
print(user_correlation)  


# In[41]:


user_correlation.shape 


# In[42]:


# using adjusted cosine
# prediction user-user
user_correlation[user_correlation<0]=0
user_correlation 


# In[43]:


# this is dot product between correlated matrix and user-item matrix 
user_predicted_ratings = np.dot(user_correlation, df_pivot.fillna(0))
user_predicted_ratings 


# In[44]:


user_predicted_ratings.shape 


# In[45]:


user_predicted_ratings 


# In[46]:


user_final_rating = np.multiply(user_predicted_ratings,dummy_train)
user_final_rating.head() 


# In[60]:


# Finding the top 5 recommendation for the user
# take the user id as input

user_input = str(input("Enter your user name"))
print(user_input) 


# In[61]:


user_final_rating.head(20) 


# In[62]:


d = user_final_rating.loc[user_input].sort_values(ascending=False)[0:5]
# = user_final_rating.sort_values(by=user_input, ascending=False)
d 


# In[66]:


# Mapping with the categories Title / Genres
categories_mapping = pd.read_csv('C:/Users/User/Desktop/ANINDITADAS_SentimentRecommendedBasedProject/sample30.csv')
categories_mapping.head()

