

import pandas as pd
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from sklearn.metrics import accuracy_score


# In[3]:


df = pd.read_csv("final_df.csv")
d = pd.read_csv("PatID#Resolution5daysAge54years-dead.csv")
h = pd.read_csv("PatID#Resolution5daysAge54years-home.csv")
dead = d['PATIENT ID']
home = h['PATIENT ID']
lst = pd.concat([dead, home], axis=0)
lst = lst.tolist()
new_df = df.loc[df['PATIENT ID'].isin(lst)]
new_df = new_df.set_index('PATIENT ID').sort_index()
new_df


# In[4]:


X = new_df
y = X.pop('MOTIVO_ALTA/DESTINY_DISCHARGE_ING')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=42)
scaler = MinMaxScaler(feature_range=(0, 1))
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


# In[5]:



# fit model no training data
model = XGBClassifier(colsample_bytree= 0.9,learning_rate= 0.2,max_depth= 4,n_estimators= 150,subsample= 0.9,reg_alpha=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[6]:


labels = list(new_df.columns)
importances = model.feature_importances_
#print(importances)
feature_imp = pd.Series(importances,index=labels).sort_values(ascending=False)
feature_imp = feature_imp.to_dict()
feature_imp = {x:y for x,y in feature_imp.items() if y!=0}
feature_imp


# In[7]:



# params = {
#         'min_child_weight': [1, 5, 10],
#         'gamma': [0.5, 1, 1.5, 2, 5],
#         'subsample': [0.6, 0.8, 1.0],
#         'colsample_bytree': [0.6, 0.8, 1.0],
#         'max_depth': [3, 4, 5]
#         }
params = {
        'n_estimators':[100,150],
        'reg_alpha':[1],
        'subsample': [0.6, 0.8, 0.9 ,1.0],
        'colsample_bytree': [0.6, 0.8, 0.9 ,1.0],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.1,0.2]
        }

xgb= XGBClassifier()
CV_rfc = GridSearchCV(estimator=xgb, param_grid=params, cv= 10)
CV_rfc = CV_rfc.fit(X_train, y_train)
params = CV_rfc.best_params_
print(params)
pred = CV_rfc.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[8]:


colsample_bytree = params['colsample_bytree']
learning_rate = params['learning_rate']
max_depth = params['max_depth']
n_estimators = params['n_estimators']
reg_alpha = params['reg_alpha']
subsample = params['subsample']
labels = list(new_df.columns)
xgb1 = XGBClassifier(colsample_bytree=colsample_bytree,learning_rate=learning_rate,max_depth=max_depth,
                     n_estimators=n_estimators,reg_alpha=reg_alpha,subsample=subsample)
xgb1.fit(X_train,y_train)
#for feature importance
importances = xgb1.feature_importances_
#print(importances)
feature_imp = pd.Series(importances,index=labels).sort_values(ascending=False)
feature_imp = feature_imp.to_dict()
feature_imp = {x:y for x,y in feature_imp.items() if y!=0}
feature_imp

