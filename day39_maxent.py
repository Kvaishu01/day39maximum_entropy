import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

st.set_page_config(page_title="Day 39 - MaxEnt Classifier", layout="wide")
st.title("📖 Day 39 — Maximum Entropy (MaxEnt) Model for Text Classification")

st.markdown("""
The **Maximum Entropy (MaxEnt)** model, also known as multinomial logistic regression, 
is a probabilistic classifier widely used in **Natural Language Processing (NLP)**.

This demo shows how to classify text documents into categories using MaxEnt.
""")

# --- Sample dataset ---
@st.cache_data
def load_sample_data():
    data = {
        "text": [
            "The government passed a new law on economy",
            "The football team won the championship",
            "New movie released in theaters",
            "Elections are coming next month",
            "The player scored two goals",
            "Critics praised the actor's performance",
            "Parliament discussed the new policy",
            "Basketball finals were exciting",
            "The actress signed a new film",
            "President announced reforms"
        ],
        "label": [
            "Politics",
            "Sports",
            "Entertainment",
            "Politics",
            "Sports",
            "Entertainment",
            "Politics",
            "Sports",
            "Entertainment",
            "Politics"
        ]
    }
    return pd.DataFrame(data)

uploaded = st.file_uploader("Upload a CSV with 'text' and 'label' columns", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
else:
    st.info("Using sample dataset (Politics, Sports, Entertainment).")
    df = load_sample_data()

st.subheader("📂 Dataset Preview")
st.dataframe(df.head())

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.3, random_state=42)

# --- Feature extraction ---
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# --- MaxEnt model ---
model = LogisticRegression(max_iter=500, multi_class="multinomial")
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

# --- Results ---
st.subheader("📊 Results")
acc = accuracy_score(y_test, y_pred)
st.write(f"**Accuracy:** {acc:.2f}")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# --- Try custom input ---
st.subheader("✍️ Try Your Own Text")
user_text = st.text_input("Enter a sentence:")
if user_text:
    vec = vectorizer.transform([user_text])
    pred = model.predict(vec)[0]
    st.success(f"Predicted Category: **{pred}**")
