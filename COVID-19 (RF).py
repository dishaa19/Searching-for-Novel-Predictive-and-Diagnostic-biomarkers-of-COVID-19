
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


ip = pd.read_csv("COVID-19.csv")
new_ip = ip.set_index('ID')


# In[3]:


new_ip


# In[4]:


Training = new_ip[new_ip['Dataset'] == 'Training']
Test = new_ip[new_ip['Dataset']== 'Test']
Training = Training.drop(columns=['Dataset'])
Test = Test.drop(columns=['Dataset'])


# In[5]:


X_train = Training.loc[:, Training.columns != 'Diagnosis']
X_test = Test.loc[:, Test.columns != 'Diagnosis']
Y_train = Training.loc[:, Training.columns == 'Diagnosis']
Y_test = Test.loc[:, Test.columns == 'Diagnosis']


# In[6]:


X_train = X_train.replace('male',1)
X_train = X_train.replace('female',0)
X_test = X_test.replace('male',1)
X_test = X_test.replace('female',0)


# In[18]:


class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns 

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)


# In[24]:


def processor(data):
    data = MultiColumnLabelEncoder(columns = ['Dataset','Diagnosis','Sex']).fit_transform(new_ip)
    
    bool_map = {True:1, False:0}

    data = data.applymap(lambda x: bool_map.get(x,x))
    
    return data


# In[25]:


from sklearn.preprocessing import LabelEncoder
df_encoded = processor(new_ip)


# In[31]:





# In[8]:


#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=1000)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,Y_train)

y_pred=clf.predict(X_test)


# In[9]:


y_pred.shape


# In[12]:


Y_test


# In[11]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))

