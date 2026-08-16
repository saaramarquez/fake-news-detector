"""
app.py
Streamlit web app for the AI Fake News Detector — custom-styled version.

Run:
    streamlit run app.py
"""

import re
import string
import joblib
import streamlit as st

st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="🛰️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------- Custom styling ----------
st.markdown(
    """
    <style>
        /* Overall page */
        .stApp {
            background: linear-gradient(180deg, #0f1117 0%, #161925 100%);
        }

        /* Main title */
        .app-title {
            font-size: 2.4rem;
            font-weight: 800;
            background: linear-gradient(90deg, #38bdf8, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0;
        }
        .app-subtitle {
            color: #9ca3af;
            font-size: 1rem;
            margin-top: 0.2rem;
            margin-bottom: 1.6rem;
        }

        /* Text area */
        .stTextArea textarea {
            background-color: #1c2030;
            color: #e5e7eb;
            border: 1px solid #2e3350;
            border-radius: 10px;
            font-size: 0.95rem;
        }

        /* Button */
        .stButton > button {
            background: linear-gradient(90deg, #38bdf8, #6366f1);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 1.6rem;
            font-weight: 600;
            width: 100%;
            transition: transform 0.15s ease;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4);
        }

        /* Result cards */
        .result-card {
            border-radius: 14px;
            padding: 1.4rem 1.6rem;
            margin-top: 1.2rem;
            border: 1px solid;
        }
        .result-real {
            background: rgba(16, 185, 129, 0.08);
            border-color: rgba(16, 185, 129, 0.4);
        }
        .result-fake {
            background: rgba(239, 68, 68, 0.08);
            border-color: rgba(239, 68, 68, 0.4);
        }
        .result-label {
            font-size: 1.5rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }
        .result-real .result-label { color: #34d399; }
        .result-fake .result-label { color: #f87171; }

        .confidence-track {
            width: 100%;
            height: 10px;
            border-radius: 6px;
            background: #2a2f45;
            overflow: hidden;
            margin-top: 0.6rem;
        }
        .confidence-fill-real {
            height: 100%;
            background: linear-gradient(90deg, #34d399, #10b981);
        }
        .confidence-fill-fake {
            height: 100%;
            background: linear-gradient(90deg, #f87171, #ef4444);
        }

        footer {visibility: hidden;}
        #MainMenu {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### 🛰️ About this project")
    st.write(
        "A machine learning model that classifies news article text as "
        "**REAL** or **FAKE**, trained with TF-IDF + Logistic Regression "
        "on a labeled news dataset."
    )
    st.markdown("---")
    st.markdown("**Model:** Logistic Regression")
    st.markdown("**Features:** TF-IDF (50,000 terms)")
    st.markdown("**Test accuracy:** ~98.6%")
    st.markdown("---")
    st.caption(
        "⚠️ Detects writing-style patterns learned from the training data — "
        "not a fact-checker. Treat results as a signal, not ground truth."
    )


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource
def load_model():
    model = joblib.load("model/fake_news_model.joblib")
    vectorizer = joblib.load("model/vectorizer.joblib")
    return model, vectorizer


# ---------- Header ----------
st.markdown('<p class="app-title">📰 AI Fake News Detector</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="app-subtitle">Paste an article\'s headline and/or body text below '
    "to check whether it looks REAL or FAKE.</p>",
    unsafe_allow_html=True,
)

try:
    model, vectorizer = load_model()
except FileNotFoundError:
    st.error(
        "Model files not found. Run `python train_model.py` first "
        "to generate model/fake_news_model.joblib and model/vectorizer.joblib."
    )
    st.stop()

user_text = st.text_area(
    "Article text",
    height=240,
    placeholder="Paste the article's headline and/or body text here...",
    label_visibility="collapsed",
)

check_clicked = st.button("🔍 Check Article")

if check_clicked:
    if not user_text.strip():
        st.warning("Please paste some text first.")
    else:
        cleaned = clean_text(user_text)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        confidence = max(proba) * 100

        if prediction == 1:
            st.markdown(
                f"""
                <div class="result-card result-real">
                    <div class="result-label">✅ Looks REAL</div>
                    <div style="color:#d1d5db;">Confidence: {confidence:.1f}%</div>
                    <div class="confidence-track">
                        <div class="confidence-fill-real" style="width:{confidence:.1f}%;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result-card result-fake">
                    <div class="result-label">🚨 Looks FAKE</div>
                    <div style="color:#d1d5db;">Confidence: {confidence:.1f}%</div>
                    <div class="confidence-track">
                        <div class="confidence-fill-fake" style="width:{confidence:.1f}%;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("See probability breakdown"):
            col1, col2 = st.columns(2)
            col1.metric("FAKE probability", f"{proba[0]*100:.1f}%")
            col2.metric("REAL probability", f"{proba[1]*100:.1f}%")

st.markdown("<br>", unsafe_allow_html=True)
st.caption(
    "⚠️ This is a machine learning demo trained on a fixed dataset — "
    "it judges writing style/patterns, not factual accuracy. "
    "Don't treat its output as ground truth."
)
