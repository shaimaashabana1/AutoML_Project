import streamlit as st

st.title(" Machine Learning Web App")
st.markdown("### Welcome to the automated Machine Learning pipeline.")
st.write("This platform helps you navigate through the entire ML lifecycle effortlessly.")

st.divider()
st.image(   "https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png"
, 
        caption="End-to-End ML Workflow", 
        use_container_width=True)

st.divider()

st.subheader(" What you can do here:")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("  Data Upload")
    st.info("Upload your CSV or Excel files and get an instant summary of your data.")

with col2:
    st.markdown("Preprocessing")
    st.success("Clean missing values, encode categorical data, and scale features automatically.")

with col3:
    st.markdown(" Analysis")
    st.warning("Visualize correlations and distributions to understand your data better.")

st.divider()

st.write("Ready to start your project?")
if st.button('Go to Upload Page '):
    st.balloons()
    st.info("Please select 'Upload' from the sidebar to begin!")
