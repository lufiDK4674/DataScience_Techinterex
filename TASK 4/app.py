import streamlit as st
import pickle
import numpy as np
from fake_news_detection import load_split_data, build_fake_news_detector, predict_news

# Load or retrain model if not available
@st.cache_resource
def get_model():
    fake_path = "News _dataset\Fake.csv"
    true_path = "News _dataset\True.csv"
    
    df = load_split_data(fake_path, true_path)
    model, vectorizer = build_fake_news_detector(df)
    return model, vectorizer

# Streamlit App UI
def main():
    st.title("📰 Fake News Detection App")
    st.write("Enter a news headline and article body below. The model will predict if it's real or fake.")

    st.markdown("---")
    
    news_title = st.text_input("📝 News Headline (optional)", "")
    news_body = st.text_area("📰 News Article Text", height=300)
    
    if st.button("🔍 Predict"):
        if news_body.strip() == "":
            st.warning("Please enter some article text to analyze.")
        else:
            with st.spinner("Analyzing..."):
                model, vectorizer = get_model()
                result = predict_news(model, vectorizer, news_body, news_title)
                
                st.subheader("🔎 Prediction Result")
                st.write(f"**Prediction:** `{result['prediction']}`")
                st.write(f"**Confidence:** `{result['confidence']*100:.2f}%`")
                
                if result['is_fake']:
                    st.error("⚠️ This article is likely FAKE.")
                else:
                    st.success("✅ This article is likely REAL.")

    st.markdown("---")

if __name__ == "__main__":
    main()
