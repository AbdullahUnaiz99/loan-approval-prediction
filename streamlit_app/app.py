import streamlit as st
from pages import prediction, model_comparison, data_insights

PAGES = {
    "🔮 Predict": prediction.render,
    "📊 Compare Models": model_comparison.render,
    "📈 Data Insights": data_insights.render,
}

st.sidebar.title("Loan Classification App")
selection = st.sidebar.radio("Navigate", list(PAGES.keys()))
PAGES[selection]()
