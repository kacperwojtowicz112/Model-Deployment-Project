import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.svm import SVR, SVC
import time


def evaluate_model_reg(actual, pred):
    mse = mean_squared_error(actual, pred)
    rmse=np.sqrt(mse)
    mae = mean_absolute_error(actual, pred)
    return mse,rmse,mae

def evaluate_model_class(actual, pred):
    acc = accuracy_score(y_true = actual, y_pred = pred) 
    sensitivity = recall_score(y_true = actual, y_pred = pred,average='macro')
    specificity = precision_score(y_true = actual, y_pred = pred,average='macro')
    return acc, sensitivity,specificity

models=["DecisionTree","RandomForest","KNN","SVM"]
def train_reg(X_train,y_train,X_test,mod):
    
    if mod=="DecisionTree":
        model = DecisionTreeRegressor()
    elif mod=="RandomForest":
        model = RandomForestRegressor()
    elif mod=="KNN":
        model = KNeighborsRegressor(n_neighbors=5)
    else:
        model = SVR(kernel='linear')
    model.fit(X_train, y_train)
    pred=model.predict(X_test)
    return pred

def train_class(X_train,y_train,X_test,mod):
    if mod=="DecisionTree":
        model = DecisionTreeClassifier()
    elif mod=="RandomForest":
        model = RandomForestClassifier()
    elif mod=="KNN":
        model = KNeighborsClassifier(n_neighbors=5)
    else:
        model = SVC(kernel='linear')
    st.write(X_train)
    model.fit(X_train, y_train)
    pred=model.predict(X_test)
    return pred

def train_opt_class(X_train,y_train,X_test,mod):
    
    if mod=="DecisionTree":
        ctriterion= st.selectbox("Select criterion:", ['gini', 'entropy'])
        max_depth=st.select_slider("Select max depth:",range(1, 15),5)
        min_split=st.select_slider("Select min samples split:",range(1, 15),5)
        min_leaf=st.select_slider("Select min samples leaf:",range(1, 15),5)
        model = DecisionTreeClassifier(
            criterion=ctriterion,
            max_depth=max_depth,
            min_samples_split=min_split, 
            min_samples_leaf=min_leaf
        )
    elif mod=="RandomForest":
        n_estimators=st.select_slider("Select number of estimators:",range(1, 100),20)
        ctriterion=st.selectbox("Select criterion:", ['gini', 'entropy'])
        max_depth=st.select_slider("Select max depth:",range(1, 15),5)
        min_split=st.select_slider("Select min samples split:",range(1, 15),5)
        min_leaf=st.select_slider("Select min samples leaf:",range(1, 15),5)
        model = RandomForestClassifier(
                n_estimators=n_estimators,
                criterion=ctriterion,
                max_depth=max_depth,
                min_samples_split=min_split, 
                min_samples_leaf=min_leaf)
    elif mod=="KNN":
        n_neighbors=st.select_slider("Select number of neighbors:",range(1, 15),5)
        weights=st.selectbox("Select weights:", ['uniform', 'distance'])
        algorithm=st.selectbox("Select algorithm:", ['auto', 'ball_tree','kd_tree', 'brute'])
        leaf_size=st.select_slider("Select leaf size:",range(1, 60),30)
        model = KNeighborsClassifier(
            n_neighbors=n_neighbors, 
            weights=weights, 
            algorithm=algorithm, 
            leaf_size=leaf_size)
    else:
        c=st.select_slider("Select c:",range(0, 5,0.2),5)
        kernel=st.selectbox("Select kernel:", ['linear', 'poly','rbf','sigmoid'])
        degree=st.select_slider("Select degree:",range(1, 10),3)
        model = SVC(C=c,kernel=kernel,degree=degree)
    if st.button("Submit",key=30):
        model.fit(X_train, y_train)
        pred=model.predict(X_test)
        return pred

def train_opt_reg(X_train,y_train,X_test,mod):
    
    if mod=="DecisionTree":
        ctriterion= st.selectbox("Select criterion:", ['absolute_error', 'squared_error', 'poisson', 'friedman_mse'])
        max_depth=st.select_slider("Select max depth:",range(1, 16),5)
        min_split=st.select_slider("Select min samples split:",range(1, 16),5)
        min_leaf=st.select_slider("Select min samples leaf:",range(1, 16),5)
        model = DecisionTreeRegressor(
            criterion=ctriterion,
            max_depth=max_depth,
            min_samples_split=min_split, 
            min_samples_leaf=min_leaf
        )
    elif mod=="RandomForest":
        n_estimators=st.select_slider("Select number of estimators:",range(1, 100),20)
        ctriterion=st.selectbox("Select criterion:", ['absolute_error', 'squared_error', 'poisson', 'friedman_mse'])
        max_depth=st.select_slider("Select max depth:",range(1, 15),5)
        min_split=st.select_slider("Select min samples split:",range(1, 15),5)
        min_leaf=st.select_slider("Select min samples leaf:",range(1, 15),5)
        model = RandomForestRegressor(
                n_estimators=n_estimators,
                criterion=ctriterion,
                max_depth=max_depth,
                min_samples_split=min_split, 
                min_samples_leaf=min_leaf)
    elif mod=="KNN":
        n_neighbors=st.select_slider("Select number of neighbors:",range(1, 15),5)
        weights=st.selectbox("Select weights:", ['uniform', 'distance'])
        algorithm=st.selectbox("Select algorithm:", ['auto', 'ball_tree','kd_tree', 'brute'])
        leaf_size=st.select_slider("Select leaf size:",range(1, 60),30)
        model = KNeighborsRegressor(
            n_neighbors=n_neighbors, 
            weights=weights, 
            algorithm=algorithm, 
            leaf_size=leaf_size)
    else:
        c=st.select_slider("Select c:",range(1, 6),3)
        kernel=st.selectbox("Select kernel:", ['linear', 'poly','rbf','sigmoid'])
        degree=st.select_slider("Select degree:",range(1, 10),3)
        model = SVR(C=c,kernel=kernel,degree=degree)
    if st.button("Submit",key=20):
        model.fit(X_train, y_train)
        pred=model.predict(X_test)
        return pred

    






#icon_image = Image.open("xdd.webp")

# st.set_page_config(
#     page_title="Model creator",
#     page_icon=icon_image,
# )
st.title("Model creator")
with st.sidebar:
    selected_option = option_menu("Main Menu", ["Choose an option","Add data", "See data","Train model"], default_index=0)

if selected_option == "Choose an option":
    
    st.write("Welcome")
elif selected_option =="Add data":
    input_file = st.file_uploader("Upload a CSV file", type="csv", accept_multiple_files=False)
    if input_file is not None:
        st.info("Success")
        if st.button("Upload"):
            st.info("Uploading...")
            df = pd.read_csv(input_file)
            df.to_csv("things/df.csv", index=False)
            st.info("Done")
elif selected_option =="See data":
    if os.path.exists("things/df.csv"):
        df = pd.read_csv("things/df.csv")
        col1,col2 = st.columns(2)
        tab_titles = ['Variable types', 'Basic statistics', 'Correlation','Plots']
        tab1, tab2, tab3,tab4 = st.tabs(tab_titles)
        with tab1:
            st.table(df.dtypes)
        with tab2:
            des=df.describe()
            st.table(des.round(2))
        with tab3:
            numeric_features = df.select_dtypes(include=['number']).columns.tolist()
            correlation_matrix = df[numeric_features].corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, fmt=".2f")
            plt.title("Correlation")
            plt.xticks(rotation=45)
            plt.yticks(rotation=45)
            st.pyplot(plt)
            plt.show()
        with tab4:
            cols=st.multiselect("Select columns:",df.columns)
            if len(cols)==1:
                selected=cols[0]
                type = df[cols[0]].dtype
                if type in ['int64', 'float64']:
                    plt.figure(figsize=(8, 6))
                    sns.histplot(df[selected], kde=True)
                    st.pyplot(plt)
                else:
                    plt.figure(figsize=(8, 6))
                    sns.countplot(x=selected, data=df)
                    plt.xticks(rotation=90)
                    st.pyplot(plt)
            elif len(cols)==2:
                selected1=cols[0]
                type1 = df[cols[0]].dtype
                selected2=cols[1]
                type2 = df[cols[1]].dtype
                if type1 in ['int64', 'float64'] and type2 in ['int64', 'float64']:
                    plt.figure(figsize=(8, 6))
                    sns.scatterplot(x=selected1, y=selected2, data=df)
                    plt.xlabel(selected1)
                    plt.ylabel(selected2)
                    plt.title(selected1 + "~" + selected2)
                    plt.grid(True)
                    st.pyplot(plt)
                elif type1 in ['int64', 'float64'] or type2 in ['int64', 'float64']:
                    if  type1 in ['int64', 'float64']:
                        plt.figure(figsize=(8, 6))
                        sns.boxplot(x=selected2, y=selected1, data=df)
                        plt.xlabel(selected2)
                        plt.ylabel(selected1)
                        plt.title(selected1 + "~" + selected2)
                        plt.grid(True)
                        st.pyplot(plt)
                    else: 
                        plt.figure(figsize=(8, 6))
                        sns.boxplot(x=selected1, y=selected2, data=df)
                        plt.xlabel(selected2)
                        plt.ylabel(selected1)
                        plt.title(selected2 + "~" + selected1)
                        plt.grid(True)
                        st.pyplot(plt)
                else:
                    t=[]
                    uni1=df[selected1].unique()
                    uni2=df[selected2].unique()
                    for i in uni1:
                        for j in uni2:
                            df1=df[df[selected1]==i]
                            df2=df1[df1[selected2]==j]
                            l=df2.shape[0]
                            t.append(l)
                    t=np.array(t)
                    matrix = t.reshape(len(uni1), len(uni2))
                    tp=pd.DataFrame(matrix,columns=uni2,index=uni1)
                    st.table(tp)
            elif len(cols)>2:
                st.info("Too many columns selected")
            else:
                st.write("")
    else:
        st.info("Add your data")
elif selected_option =="Train model":
    if os.path.exists("things/df.csv"):
        tab_titles=["Compare entry-level models","Train model"]
        tab1, tab2 = st.tabs(tab_titles)
        df = pd.read_csv("things/df.csv")
        
        with tab1:
            ycol=st.selectbox("Select Y:",df.columns,key="key1")
            models=["DecisionTree","RandomForest","KNN","SVM"]
            mod=st.multiselect("Select models to compare",models)
            y=df[ycol]
            X=df.drop(ycol,axis=1)
            lentest=st.select_slider("What part of the set should be the test set?",range(1, 100),20,key="key11")
            X_dummies = pd.get_dummies(X)
            X_train, X_test,y_train, y_test = train_test_split(X_dummies,y, test_size=lentest/100,random_state = 2115)
            if st.button("Submit",key=10):
                start_time = time.time()
                if y.dtype in ['int64', 'float64']:
                    t=pd.DataFrame(index=["mse","rmse","mae"])
                    for i in mod:
                        pred = train_reg(X_train,y_train,X_test,i)
                        mse,rmse,mae= evaluate_model_reg(y_test,pred)
                        t[i]=[mse,rmse,mae]

                else:
                    t=pd.DataFrame(index=["acc", "sensitivity","specificity"])
                    for i in models:
                        pred = train_class(X_train,y_train,X_test,i)
                        acc, sensitivity,specificity= evaluate_model_class(y_test,pred)
                        t[i]=[acc, sensitivity,specificity]
                st.table(t)
                end_time = time.time()
                training_time_seconds = end_time - start_time

                st.write("Training time:", training_time_seconds, "seconds")

        with tab2:
            
            ycol=st.selectbox("Select Y:",df.columns,key="key2")
            y=df[ycol]
            X=df.drop(ycol,axis=1)
            X_dummies = pd.get_dummies(X)
            models=["DecisionTree","RandomForest","KNN","SVM"]
            mod=st.selectbox("Select model:", models)
            lentest=st.select_slider("What part of the set should be the test set?",range(1, 100),20,key="key21")
            X_train, X_test, y_train, y_test = train_test_split(X_dummies, y, test_size=lentest,random_state = 2115)
            if y.dtype in ['int64', 'float64']:
                pred = train_opt_reg(X_train,y_train,X_test,mod)
                if pred is not None:
                    mse,rmse,mae= evaluate_model_reg(y_test,pred)
                    st.write("MSE:",mse)
                    st.write("RMSE:",rmse)
                    st.write("MAE:",mae)
            else: 
                pred = train_opt_class(X_train,y_train,X_test,mod)
                if pred is not None:
                    acc, sensitivity,specificity= evaluate_model_class(y_test,pred)
                    st.write("Accuracy:",acc)
                    st.write("Sensiticity:",sensitivity)
                    st.write("Specificity:",specificity)

    else:
        st.info("Add your data")
    