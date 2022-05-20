#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd  
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report 
from sklearn.metrics import confusion_matrix


# In[2]:


cardio_f = pd.read_csv("cardio_train.csv")
# cardio_f.head()


# In[5]:


feature_f = cardio_f [['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
       'cholesterol', 'gluc', 'smoke', 'alco', 'active']]

x = np.asarray(feature_f)

y = np.asarray(cardio_f['cardio'])


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.9, random_state = 0)

'''
train sample: 6930
test sample: 62371
'''

# In[8]:
 

classifier = svm.SVC(kernel = 'linear', gamma = 'auto', C = 1)
classifier.fit(X_train, y_train)

y_predict = classifier.predict(X_test)

# In[12]:


# print(classification_report(y_test, y_predict))
print('Predicted labels: ', y_predict[0:10])
print('Accuracy test: ', accuracy_score(y_test, y_predict))

# In[ ]:

