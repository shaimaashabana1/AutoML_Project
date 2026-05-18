import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                             mean_absolute_error, r2_score, silhouette_score)

st.title(" Model Evaluation")

if "model_ready" not in st.session_state:
    st.info("No model trained yet. Please go to Modeling page.")
    st.stop()

res = st.session_state["model_ready"]

if res["task"] == "Classification":
    y_pred = res["model"].predict(res["X_test"])
    
    acc = accuracy_score(res["y_test"], y_pred)
    st.metric("Overall Accuracy", f"{acc*100:.2f}%")
    
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(res["y_test"], y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    st.pyplot(fig)
    
    # Report
    st.subheader("Detailed Report")
    st.text(classification_report(res["y_test"], y_pred))

elif res["task"] == "Regression":
    y_pred = res["model"].predict(res["X_test"])
    st.metric("R2 Score", round(r2_score(res["y_test"], y_pred), 4))
    st.metric("MAE", round(mean_absolute_error(res["y_test"], y_pred), 4))
    
    fig, ax = plt.subplots()
    plt.scatter(res["y_test"], y_pred, alpha=0.5)
    plt.plot([res["y_test"].min(), res["y_test"].max()], [res["y_test"].min(), res["y_test"].max()], 'r--')
    st.pyplot(fig)

elif res["task"] == "Clustering":
    score = silhouette_score(res["X"], res["clusters"])
    st.metric("Silhouette Score", round(score, 4))
    
    fig, ax = plt.subplots()
    plt.scatter(res["X"].iloc[:, 0], res["X"].iloc[:, 1], c=res["clusters"], cmap='viridis')
    st.pyplot(fig)
    