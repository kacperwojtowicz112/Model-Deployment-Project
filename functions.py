
import numpy as np
import streamlit as st
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, recall_score, precision_score,classification_report
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.model_selection import GridSearchCV


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
    
def opt_class(X_train,y_train,X_test,mod):
    
    if mod=="DecisionTree":
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': range(1, 30,2),
            'min_samples_split': range(1, 25,2),
            'min_samples_leaf': range(1, 15,2),
            'max_features': [None, 'auto', 'sqrt', 'log2']
        }

        model = DecisionTreeClassifier()

        grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        st.write(grid_search.best_params_)

        best_model = grid_search.best_estimator_
        pred = best_model.predict(X_test)
    elif mod=="RandomForest":
        param_grid = {
            'n_estimators': range(1, 100,4),
            'criterion': ['gini', 'entropy'],
            'max_depth': range(1, 30,2),
            'min_samples_split': range(1, 30,2),
            'min_samples_leaf': range(1, 15,2),
            'max_features': ['auto', 'sqrt', 'log2']
        }

        model = RandomForestClassifier()

        grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        st.write(grid_search.best_params_)

        best_model = grid_search.best_estimator_
        pred = best_model.predict(X_test)
    elif mod=="KNN":
        param_grid = {
            'n_neighbors': range(1, 15,2),
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': range(1, 50,4),
            'p': [1, 2]
        }

        model = KNeighborsClassifier()

        grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        st.write(grid_search.best_params_)

        best_model = grid_search.best_estimator_
        pred = best_model.predict(X_test)
    else:
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': ['scale', 'auto'],
            'degree': range(2, 5),
            'coef0': [0.0, 0.1, 0.5, 1.0]
        }

        model = SVC()

        grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        st.write(grid_search.best_params_)

        best_model = grid_search.best_estimator_
        pred = best_model.predict(X_test)
    return pred
    
def opt_reg(X_train,y_train,X_test,mod):
    
    if mod=="DecisionTree":
        param_grid = {
            'criterion': ['mse', 'friedman_mse', 'mae'],
            'max_depth': range(1, 30,2),
            'min_samples_split': range(1, 25,2),
            'min_samples_leaf': range(1, 15,2),
            'max_features': [None, 'auto', 'sqrt', 'log2']
        }

        model = DecisionTreeRegressor()

        grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)

        st.write(grid_search.best_params_)

        best_model = grid_search.best_estimator_
        pred = best_model.predict(X_test)
    elif mod=="RandomForest":
        param_grid = {
            'n_estimators': range(1, 100,4),
            'criterion': ['mse', 'mae'],
            'max_depth': range(1, 30,2),
            'min_samples_split': range(1, 30,2),
            'min_samples_leaf': range(1,15,2),
            'max_features': [None, 'auto', 'sqrt', 'log2']
        }

        model = RandomForestRegressor()

        grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)

        st.write(grid_search.best_params_)

        best_model = grid_search.best_estimator_
        pred = best_model.predict(X_test)
    elif mod=="KNN":
        param_grid = {
            'n_neighbors': range(1, 10,2),
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': range(1, 50,4),
            'p': [1, 2]
        }

        model = KNeighborsRegressor()

        grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)

        st.write(grid_search.best_params_)

        best_model = grid_search.best_estimator_
        pred = best_model.predict(X_test)
    else:
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': ['scale', 'auto'],
            'degree': range(2, 5),
            'coef0': [0.0, 0.1, 0.5, 1.0]
        }

        model = SVR()

        grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)

        st.write(grid_search.best_params_)

        best_model = grid_search.best_estimator_
        pred = best_model.predict(X_test)
    return pred