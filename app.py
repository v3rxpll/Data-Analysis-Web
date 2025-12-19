import streamlit as st
import pandas as pd
from numpy.random import default_rng as rng
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

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

st.set_page_config(page_title="Linear Regression", layout="wide")
st.title("📈 Linear Regression: Predicted vs Actual")

df = pd.read_csv('sustainable_waste_management_dataset_2024.csv')
st.subheader("Dataset Preview")
st.dataframe(df.head())

selected_features = ['recyclable_kg', 'waste_kg', 'collection_capacity_kg', 'temp_c']
X = df[selected_features]
y = df['waste_kg']

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

st.write("Mean Squared Error (MSE):", mean_squared_error(Y_test, Y_pred))
st.write("R squared (R²):", r2_score(Y_test, Y_pred))

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(Y_test, Y_pred, alpha=0.7)
ax.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='red', lw=2, label='Perfect Prediction Line')
ax.set_xlabel('Actual Waste (Y_test)')
ax.set_ylabel('Predicted Waste (Y_pred)')
ax.set_title('Predicted vs Actual Waste')
ax.legend()
ax.grid(True)

st.pyplot(fig)

