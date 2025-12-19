import streamlit as st
import pandas as pd
from numpy.random import default_rng as rng
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="My Streamlit App",
    page_icon="🌱",
    layout="wide"
)

st.markdown(
    "<h1 style='text-align:center; color:#2E8B57;'>🌍 Sustainable Waste Dashboard</h1>",
    unsafe_allow_html=True
)
st.markdown("---")

st.sidebar.header("👤 User Info")
name = st.sidebar.text_input("Your name")
major = st.sidebar.selectbox("Which major do you like best?", ["CO", "CI"])

if name:
    st.sidebar.success(f"Hello {name} 👋")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Dataset Preview")
    df = pd.read_csv("sustainable_waste_management_dataset_2024.csv")
    st.dataframe(df, use_container_width=True)

with col2:
    st.subheader("📊 Random Data Visualization")
    df_random = pd.DataFrame(
        rng(0).standard_normal((20, 3)),
        columns=["a", "b", "c"]
    )
    st.bar_chart(df_random)
    st.line_chart(df_random)

st.markdown("---")
if st.button("🚀 Click me"):
    st.success("You clicked the button!")