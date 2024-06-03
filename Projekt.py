import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import time
import functions as f



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
                        pred = f.train_reg(X_train,y_train,X_test,i)
                        mse,rmse,mae= f.evaluate_model_reg(y_test,pred)
                        t[i]=[mse,rmse,mae]

                else:
                    t=pd.DataFrame(index=["acc", "sensitivity","specificity"])
                    for i in models:
                        pred = f.train_class(X_train,y_train,X_test,i)
                        acc, sensitivity,specificity= f.evaluate_model_class(y_test,pred)
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
                pred = f.train_opt_reg(X_train,y_train,X_test,mod)
                if pred is not None:
                    mse,rmse,mae= f.evaluate_model_reg(y_test,pred)
                    st.write("MSE:",mse)
                    st.write("RMSE:",rmse)
                    st.write("MAE:",mae)
            else: 
                pred = f.train_opt_class(X_train,y_train,X_test,mod)
                if pred is not None:
                    acc, sensitivity,specificity= f.evaluate_model_class(y_test,pred)
                    st.write("Accuracy:",acc)
                    st.write("Sensiticity:",sensitivity)
                    st.write("Specificity:",specificity)

    else:
        st.info("Add your data")
    