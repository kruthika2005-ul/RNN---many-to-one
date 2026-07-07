# app.py
# SMS Spam Detection using Simple RNN (Many-to-One)

import os
import re
import pickle
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==========================================================
# Configuration
# ==========================================================

MODEL = "spam_model.keras"
TOKENIZER = "tokenizer.pkl"
DATASET = "spam.csv"

MAX_WORDS = 5000
MAX_LEN = 50


# ==========================================================
# Text Cleaning
# ==========================================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ==========================================================
# Train Model
# ==========================================================

def train_model():

    print("Loading dataset...")

    if not os.path.exists(DATASET):
        st.error(f"Dataset '{DATASET}' not found.")
        st.stop()

    df = pd.read_csv(DATASET, encoding="latin-1")

    # Keep only required columns
    df = df[["v1", "v2"]]
    df.columns = ["label", "text"]

    # Convert labels
    df["label"] = df["label"].map({
        "ham": 0,
        "spam": 1
    })

    # Clean text
    df["text"] = df["text"].apply(clean_text)

    # Tokenizer
    tokenizer = Tokenizer(
        num_words=MAX_WORDS,
        oov_token="<OOV>"
    )

    tokenizer.fit_on_texts(df["text"])

    sequences = tokenizer.texts_to_sequences(df["text"])

    X = pad_sequences(
        sequences,
        maxlen=MAX_LEN,
        padding="post"
    )

    y = df["label"]

    # Save tokenizer
    with open(TOKENIZER, "wb") as f:
        pickle.dump(tokenizer, f)

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # ======================================================
    # Build Model
    # ======================================================

    model = Sequential()

    model.add(
        Embedding(
            input_dim=MAX_WORDS,
            output_dim=128,
            input_length=MAX_LEN,
        )
    )

    model.add(SimpleRNN(128))

    model.add(Dense(32, activation="relu"))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    print("Training model...")

    history = model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=1,
    )

    # Save model
    model.save(MODEL)

    # Evaluation
    loss, accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0,
    )

    print("\nAccuracy :", accuracy)

    predictions = (
        model.predict(X_test, verbose=0) > 0.5
    ).astype(int)

    print("\nClassification Report")
    print(classification_report(y_test, predictions))

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, predictions))

    print("\nTraining completed successfully.")


# ==========================================================
# Prediction Function
# ==========================================================

def predict_sms(message):

    model = load_model(MODEL)

    with open(TOKENIZER, "rb") as f:
        tokenizer = pickle.load(f)

    message = clean_text(message)

    sequence = tokenizer.texts_to_sequences([message])

    sequence = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
    )

    probability = model.predict(
        sequence,
        verbose=0,
    )[0][0]

    if probability >= 0.5:
        return " Spam", probability
    else:
        return " Ham", 1 - probability


# ==========================================================
# Train Model Only Once
# ==========================================================

if (
    not os.path.exists(MODEL)
    or not os.path.exists(TOKENIZER)
):
    train_model()


# ==========================================================
# Streamlit UI
# ==========================================================

st.set_page_config(
    page_title="SMS Spam Detector",
    layout="centered",
)

st.title("SMS Spam Detection")
st.write("### Simple RNN (Many-to-One)")

message = st.text_area(
    "Enter SMS Message",
    height=150,
)

if st.button("Predict"):

    if message.strip() == "":
        st.warning("Please enter a message.")
    else:

        prediction, confidence = predict_sms(message)

        if "Spam" in prediction:
            st.error(prediction)
        else:
            st.success(prediction)

        st.write(
            f"**Confidence:** {confidence*100:.2f}%"
        )

st.markdown("---")
st.caption("Developed using TensorFlow, Keras and Streamlit")