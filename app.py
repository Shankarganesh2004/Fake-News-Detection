import streamlit as st
import joblib

# Load vectorizer and model
vectorizer = joblib.load('vectorizer.jb')
model = joblib.load("pa_model.jb")

# App title
st.title("📰 Fake News Detection")
st.write("Enter a news article below to check whether it is Fake or Real.")

# Text input
news_input = st.text_area("News Article:", "")

if st.button("Check News"):
    if news_input.strip():
        transform_input = vectorizer.transform([news_input])
        prediction = model.predict(transform_input)

        if prediction[0] == 1:  # change to 'REAL' if your model returns strings
            st.success("✅ The News is Real")
        else:
            st.error("❌ The News is Fake")
    else:
        st.warning("⚠️ Please enter some news to analyze.")
