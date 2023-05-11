#!/usr/bin/env python
# coding: utf-8

# # SERVO PREDICTION PREDICTION MODEL 
# 
# ### YBI DATA SCIENCE AND ML INTERNSHIP
# 
# #### CREATED BY @ TARANDEEP SINGH GUJRAL 

# #### IMPORTING LIBRARIES
# 

# In[41]:


import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error 


# In[11]:


data = pd.read_csv('https://raw.githubusercontent.com/YBIFoundation/Dataset/main/Servo%20Mechanism.csv')


# In[12]:


data.head()


# In[13]:


data.info()


# In[14]:


data.describe()


# In[15]:


data.columns


# In[19]:


data.shape


# In[22]:


data[['Motor']].value_counts()


# In[23]:


data[['Screw']].value_counts()


# ### Encoding of Categorical Features

# In[26]:


data.replace({'Motor':{'A':0,'B':1,'C':2,'D':3,'E':4}},inplace=True)
data.replace({'Screw':{'A':0,'B':1,'C':2,'D':3,'E':4}},inplace=True)


# In[28]:


data.head()


# In[29]:


y = data['Class']
x = data.drop('Class',axis=1)


# ### Train Test Split

# In[36]:


xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size = 0.4, random_state=None)


# ### Model Training

# In[39]:


lmodel = LinearRegression()

lmodel.fit(xtrain,ytrain)


# ### Model Prediction

# In[40]:


ypred = lmodel.predict(xtest)
ypred


# ### Model Evaluation

# In[43]:


mean_squared_error(ytest, ypred)


# In[44]:


mean_absolute_error(ytest, ypred)


# In[45]:


r2_score(ytest, ypred)


# In[ ]:




