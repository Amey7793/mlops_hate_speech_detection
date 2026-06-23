"""
Streamlit UI for hate speech detection.
Calls the Flask API at localhost:8000.
"""
import os
import requests
import streamlit as st

FLASK_URL = os.getenv("FLASK_API_URL", "http://localhost:8000/predict")

st.set_page_config(page_title="Hate Speech Detector", page_icon="🔍")
st.title("Hate Speech Detector")
st.write("Enter a tweet or piece of text to classify it as Hate Speech or Not Hate Speech.")

text_input = st.text_area("Input Text", height=120, placeholder="Type your text here...")

if st.button("Predict", type="primary"):
    if not text_input.strip():
        st.warning("Please enter some text before predicting.")
    else:
        with st.spinner("Classifying..."):
            try:
                response = requests.post(FLASK_URL, json={"text": text_input}, timeout=10)
                response.raise_for_status()
                result = response.json()

                st.subheader("Result")

                label = result["label"]
                confidence = result["confidence"]
                color = "red" if label == "Hate Speech" else "green"

                st.markdown(f"### :{color}[{label}]")
                st.metric("Confidence", f"{confidence:.2%}")
                st.caption(f"Model: {result['model']}")

                with st.expander("Cleaned Text"):
                    st.write(result["cleaned"])

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to Flask API. Make sure it is running on localhost:8000.")
            except requests.exceptions.HTTPError as e:
                st.error(f"API error: {e}")
