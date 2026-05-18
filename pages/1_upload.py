import streamlit as st
import pandas as pd

st.title("File Upload")

file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])

if file:
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    st.session_state["data"] = df

    st.success("File loaded successfully")

    st.subheader("Preview")
    st.dataframe(df.head())

    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])

else:
    st.info("Please upload a CSV or Excel file")