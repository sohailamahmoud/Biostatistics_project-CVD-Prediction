#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd  
import numpy as np
import pickle

# In[2]:


cardio_f = pd.read_csv("cardio_train.csv")
cardio_f.head()
cardio_f.tail()
cardio_f.size
cardio_f.count()
cardio_f['cardio'].value_counts()


# In[3]:


positive_f = cardio_f[cardio_f['cardio']==1] [0:200]
negative_f = cardio_f[cardio_f['cardio']==0] [0:200]


# In[4]:


cardio_f.dtypes



# In[5]:


cardio_f.columns

feature_f = cardio_f [['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
       'cholesterol', 'gluc', 'smoke', 'alco', 'active']]

x = np.asarray(feature_f)

y = np.asarray(cardio_f['cardio'])

y[0:5]


# In[7]:



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.8, random_state = 4)

# (13860, 11)
X_train.shape

# (13860, 1)
y_train.shape

# (55441, 11)
X_test.shape

# (55441, 1)
y_test.shape


# In[8]:


from sklearn import svm

classifier = svm.SVC(kernel = 'linear', gamma = 'auto', C = 1)
classifier.fit(X_train, y_train)

y_predict = classifier.predict(X_test)


# In[12]:


from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, classification_report 


print(classification_report(y_test, y_predict))
print('Predicted labels: ', y_predict)
print('Accuracy: ', accuracy_score(y_test, y_predict))


# In[ ]:

pickle.dump(cardio_f, open('finalized_model.sav', 'wb'))

# %%
