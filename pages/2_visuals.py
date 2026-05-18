import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Data Visualization")

if "data" not in st.session_state:
    st.warning("Please upload data first")
    st.stop()

df = st.session_state["data"]
columns = df.columns

plot_type = st.selectbox(
    "Select Plot Type",
    ["Line Plot", "Scatter Plot", "Box Plot"]
)

fig, ax = plt.subplots()

if plot_type == "Box Plot":
    selected_cols = st.multiselect("Select columns for Box Plot", columns)
    if selected_cols:
        sns.boxplot(data=df[selected_cols], ax=ax)
    else:
        st.info("Please select at least one column")

elif plot_type == "Line Plot":
    x_col = st.selectbox("X axis", columns, key="lx")
    y_col = st.selectbox("Y axis", columns, key="ly")
    sns.lineplot(data=df, x=x_col, y=y_col, ax=ax)

elif plot_type == "Scatter Plot":
    x_col = st.selectbox("X axis", columns, key="sx")
    y_col = st.selectbox("Y axis", columns, key="sy")
    sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)

# عرض الرسمة لو فيه داتا مرسومة
if plot_type != "Box Plot" or selected_cols:
    st.pyplot(fig)