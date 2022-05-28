#!/usr/bin/env python
# coding: utf-8

# In[1]:

# from hashlib import sha1
import pandas as pd  
import numpy as np
import pickle

# In[2]:


df = pd.read_csv("cardio_train.csv")
df.head()


# In[4]:

df.shape
# (69301, 13)

# In[]:

df.dtypes

# In[]:

df.columns

# df = cardio_f [['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
#        'cholesterol', 'gluc', 'smoke', 'alco', 'active']]


# In[]:

df.age = round(df.age/ 365)


# In[]:

df.describe()

# In[]:

# Removing height outliers

height_upper_limit = df.height.mean() + 4*df.height.std()   
# height_upper_limit = 197.2

height_lower_limit = df.height.mean() - 4*df.height.std()
# height_lower_limit = 131.5

df[(df.height > height_upper_limit) | (df.height < height_lower_limit)].shape
# 121 outlier    (121, 13)

df = df[(df.height < height_upper_limit) & (df.height > height_lower_limit)]

df.shape
# (69180, 13)

# In[]:

# Removing weight outliers

df[(df.weight > 200) | (df.weight < 40)].shape

df = df[(df.weight < 200) & (df.weight > 40)]

df.shape
# (69087, 13)

# In[]:

df.describe()


# In[]:

# Removing ap_hi outliers

df[(df.ap_hi > 370) | (df.ap_hi < 50)].shape
# (224, 13)

df = df[(df.ap_hi < 370) & (df.ap_hi > 50)]

df.shape
# (68863, 13)


# In[]:

# Removing ap_lo outliers


df[(df.ap_lo > 360) | (df.ap_lo < 20)].shape

df = df[(df.ap_lo < 360) & (df.ap_lo > 20)]

df.shape
# (67887, 13)

# In[]:

df_model = df [['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
       'cholesterol', 'gluc', 'smoke', 'alco', 'active']]

df_model.describe()




# In[6]:

x = np.asarray(df_model)

y = np.asarray(df['cardio'])

y[0:5]


# In[7]:

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 4)

# (13577, 11)
X_train.shape

# (13577, 1)
y_train.shape

# (54310, 11)
X_test.shape

# (54310, 1)
y_test.shape


# In[8]:


from sklearn import svm

classifier = svm.SVC(kernel = 'linear', gamma = 'auto', C = 1)
classifier.fit(X_train, y_train)

y_predict = classifier.predict(X_test)


# In[12]:


from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, classification_report 


# print(classification_report(y_test, y_predict))
print('Predicted labels: ', y_predict[0:10])
print('Accuracy: ', accuracy_score(y_test, y_predict))

pickle.dump(classifier, open("model.pkl","wb"))

# In[ ]:



# %%
