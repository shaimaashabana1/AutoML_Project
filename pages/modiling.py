import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier

st.title(" Model Selection")

if "data" not in st.session_state:
    st.warning("Please upload and preprocess data first!")
    st.stop()

df = st.session_state["data"]

task_type = st.selectbox("Select Learning Type", 
                        ["Classification (Supervised)", 
                         "Regression (Supervised)", 
                         "Clustering (Unsupervised)"])

target = None
if task_type != "Clustering (Unsupervised)":
    target = st.selectbox("Select Target Variable", df.columns)
    X = df.drop(columns=[target]).select_dtypes(include=[np.number])
    y = df[target]
else:
    X = df.select_dtypes(include=[np.number])

model = None
if task_type == "Classification (Supervised)":
    algo = st.selectbox("Choose Algorithm", 
        ["Decision Tree", "Logistic Regression", "SVM", "Random Forest", "KNN", "Bayesian Classifier", "Neural Networks"])
    
    if algo == "Decision Tree": model = DecisionTreeClassifier()
    elif algo == "Logistic Regression": model = LogisticRegression(max_iter=1000)
    elif algo == "SVM": model = SVC(probability=True)
    elif algo == "Random Forest": model = RandomForestClassifier()
    elif algo == "KNN": model = KNeighborsClassifier()
    elif algo == "Bayesian Classifier": model = GaussianNB()
    elif algo == "Neural Networks": model = MLPClassifier(max_iter=500)

elif task_type == "Regression (Supervised)":
    st.info("Algorithm: Linear Regression")
    model = LinearRegression()

elif task_type == "Clustering (Unsupervised)":
    st.subheader("Determine Optimal K (Elbow Method)")
    
    if st.button(" Show Elbow Curve"):
        wcss = [] 
        K_range = range(1, 11)
        for k in K_range:
            kmeans_temp = KMeans(n_clusters=k, random_state=42)
            kmeans_temp.fit(X)
            wcss.append(kmeans_temp.inertia_)
        
        fig, ax = plt.subplots()
        plt.plot(K_range, wcss, 'bx-')
        plt.xlabel('Number of clusters (K)')
        plt.ylabel('WCSS (Inertia)')
        plt.title('The Elbow Method showing the optimal k')
        st.pyplot(fig)

    k_val = st.slider("Select K (Clusters)", 2, 10, 3)
    model = KMeans(n_clusters=k_val)

if st.button(" Train & Save Model"):
    if task_type != "Clustering (Unsupervised)":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        st.session_state["model_ready"] = {
            "model": model, "X_test": X_test, "y_test": y_test, 
            "task": "Classification" if task_type.startswith("Class") else "Regression"
        }
    else:
        clusters = model.fit_predict(X)
        st.session_state["model_ready"] = {
            "model": model, "X": X, "clusters": clusters, "task": "Clustering"
        }
    
    st.success("Model Trained Successfully! Go to Evaluation Page.")