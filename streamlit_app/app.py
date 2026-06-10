import streamlit as st
import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)

# -----------------------------
# Page Config
# -----------------------------

st.set_page_config(
    page_title="Hate Speech Detection",
    page_icon="🛡️",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------

MODEL_NAME = "nsaichandana/hate-speech-weighted-distilbert"

@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        MODEL_NAME
    )

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME
    )

    return tokenizer, model

tokenizer, model = load_model()

# -----------------------------
# Label Mapping
# -----------------------------

label_map = {
    0: "Hate Speech",
    1: "Offensive Language",
    2: "Neither"
}

# -----------------------------
# UI
# -----------------------------

st.title("🛡️ Hate Speech Detection")

st.markdown(
    """
    Detect whether a social media post contains:

    - Hate Speech
    - Offensive Language
    - Neither
    """
)

tweet = st.text_area(
    "Enter Text",
    height=150,
    placeholder="Type a tweet or social media post..."
)

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict"):

    if tweet.strip() == "":
        st.warning("Please enter some text.")
    else:

        inputs = tokenizer(
            tweet,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0,
            predicted_class
        ].item()

        prediction = label_map[predicted_class]

        st.subheader("Prediction")

        if prediction == "Hate Speech":
            st.error(prediction)

        elif prediction == "Offensive Language":
            st.warning(prediction)

        else:
            st.success(prediction)

        st.write(
            f"**Confidence:** {confidence*100:.2f}%"
        )

        st.subheader("Class Probabilities")

        st.write(
            f"**Hate Speech:** {probabilities[0][0].item()*100:.2f}%"
        )

        st.write(
            f"**Offensive Language:** {probabilities[0][1].item()*100:.2f}%"
        )

        st.write(
            f"**Neither:** {probabilities[0][2].item()*100:.2f}%"
        )
