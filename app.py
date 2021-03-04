
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn import datasets


# In[2]:


st.title("Machine Learning algorithms performance and visualiztion app using deafult datasets")


# In[3]:


algo_opt=st.sidebar.selectbox("Choose algorithm",("Classification","Regression"))


# In[4]:


st.write(f"You have opted for {algo_opt}")


# In[5]:


def select_algo(algo_opt):
    if algo_opt=="Classification":
        algo=st.sidebar.selectbox("Algorithms available for classification",("KNN","Random Forest","SVM","Neural Network","Logistic Regression","Decision Tree"))
    else:
        algo=st.sidebar.selectbox("Algorithms available for regression",("Linear Regression","Random Forest Regressor","Decision Tree Regressor"))
    return algo
    


# In[6]:


algo_select=select_algo(algo_opt)


# In[7]:


st.write(f"{algo_opt} algorithm selected : {algo_select}")


# In[8]:


def select_dataset(algo_opt):
    if algo_opt=='Classification':
        data=st.sidebar.selectbox("Select the classification dataset",("Iris","Digits","Wine","Breast Cancer"))
    else:
        data=st.sidebar.selectbox("Select the regression dataset",("Boston","Diabetes"))
    return data

def data_split(data):
    if data=='Iris':
        data_load=datasets.load_iris()
    elif data=='Digits':
        data_load=datasets.load_digits()
    elif data=='Wine':
        data_load=datasets.load_wine()
    elif data=='Breast Cancer':
        data_load=datasets.load_breast_cancer()
    elif data=='Boston':
        data_load=datasets.load_boston()
    elif data=='Diabetes':
        data_load=datasets.load_diabetes()
    X=data_load.data
    y=data_load.target
    return data_load,X,y


# In[9]:


data=select_dataset(algo_opt)


# In[10]:


st.write(f"Dataset selected : {data}")


# In[11]:


data_load,X,y=data_split(data)


# In[12]:


if data == 'Digits':
    st.write("Digits dataset has no feature names")
else:
    st.write(data_load.feature_names)
st.write(f"Data shape {X.shape}")
st.write(f"Number of classes {len(np.unique(y))}")


# In[13]:


def param_select(algo_select):
    params=dict()
    if algo_select=='KNN':
        K=st.sidebar.slider("k",min_value=1,max_value=15)
        params['K']=K
    elif algo_select=='Random Forest' or algo_select=='Random Forest Regressor':
        max_depth=st.sidebar.slider("max_depth",min_value=1,max_value=32)
        n_estimators=st.sidebar.slider("n_estimators",min_value=2,max_value=100)
        params['max_depth']=max_depth
        params['n_estimators']=n_estimators
    elif algo_select=='SVM':
        C=st.sidebar.slider("C",min_value=1.0,max_value=10.0)
        params['C']=C
    elif algo_select=='Neural Network':
        layers=st.sidebar.slider("Number of layers",min_value=1,max_value=100)
        activation=st.sidebar.selectbox("Activation Function",("identity","tanh","relu","logitstic"))
        batch_size=st.sidebar.selectbox("Batch Size",("16","32","64"))
        params['layers']=layers
        params['activation']=activation
        params['batch_size']=int(batch_size)
    return params
    


# In[14]:


params=param_select(algo_select)


# In[15]:


def load_model(algo_select,params):
    if algo_select=='KNN':
        model=KNeighborsClassifier(params['K'])
    elif algo_select=='Random Forest':
        model=RandomForestClassifier(n_estimators=params['n_estimators'],max_depth=params['max_depth'],random_state=1234)
    elif algo_select=='Random Forest Regressor':
        model=RandomForestRegressor(n_estimators=params['n_estimators'],max_depth=params['max_depth'],random_state=1234)
    elif algo_select=='SVM':
        model=SVC(C=params['C'])
    elif algo_select=='Neural Network':
        model=MLPClassifier(hidden_layer_sizes=params['layers'],activation=params['activation'],batch_size=params['batch_size'])
    elif algo_select=='Logistic Regression':
        model=LogisticRegression()
    elif algo_select=='Decision Tree':
        model=DecisionTreeClassifier()
    elif algo_select=='Decision Tree Regressor':
        model=DecisionTreeRegressor()
    elif algo_select=='Linear Regression':
        model=LinearRegression()
    return model


# In[16]:


model=load_model(algo_select,params)


# In[17]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1234)


# In[18]:


model.fit(X_train,y_train)


# In[19]:


y_pred=model.predict(X_test)


# In[25]:


from sklearn.metrics import accuracy_score,f1_score,mean_squared_error,r2_score
if algo_opt=='Classification':
    st.write(f"Accuracy Score : {accuracy_score(y_test,y_pred)}")
    st.write(f"F-measure : {f1_score(y_test,y_pred,average='weighted')}")
elif algo_opt=='Regression':
    st.write(f"Mean Squared Error : {mean_squared_error(y_test,y_pred)}")
    st.write(f"R2 score : {r2_score(y_test,y_pred)}")

