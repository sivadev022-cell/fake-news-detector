import streamlit as st
import pickle
import re
from datetime import datetime

# -------------------------------
# 🔒 Load Model
# -------------------------------
@st.cache_resource
def load_model():
    try:
        return pickle.load(open("model.pkl", "rb"))
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

# -------------------------------
# 🧹 Clean Text
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -------------------------------
# 🎨 Page Config
# -------------------------------
st.set_page_config(
    page_title="Fake News Detector Pro",
    page_icon="🧠",
    layout="wide"
)

# -------------------------------
# 🌈 Styling
# -------------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}
.stTextArea textarea {
    border-radius: 10px;
    padding: 10px;
}
.stButton>button {
    border-radius: 10px;
    height: 45px;
    width: 100%;
    font-weight: bold;
}
.metric-box {
    padding: 15px;
    border-radius: 10px;
    background-color: #1f2937;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# 🧠 HEADER
# -------------------------------
st.title("🧠 Fake News Detector Pro")
st.caption("🚀 NLP-powered system to detect misinformation in real-time")

# -------------------------------
# 📊 SIDEBAR
# -------------------------------
st.sidebar.title("⚙️ Settings")

show_details = st.sidebar.checkbox("Detailed Analysis", True)
show_example = st.sidebar.checkbox("Load Example", True)

st.sidebar.markdown("---")
st.sidebar.info("""
Model: Logistic Regression  
Vectorizer: TF-IDF  
Accuracy: ~99.4%  
""")

# -------------------------------
# 📰 INPUT
# -------------------------------
col1, col2 = st.columns([3,1])

with col1:
    user_input = st.text_area("📰 Enter News Text", height=200)

with col2:
    if show_example:
        if st.button("📌 Example"):
            user_input = "Government announces new economic reforms to boost growth."

    st.write("### ℹ️ Tips")
    st.write("- Enter full news text")
    st.write("- Avoid very short sentences")

# -------------------------------
# 🔍 ANALYZE
# -------------------------------
if st.button("🔍 Analyze News"):

    if model is None:
        st.error("Model not loaded.")
    
    elif user_input.strip() == "":
        st.warning("Please enter news text.")
    
    else:
        try:
            cleaned = clean_text(user_input)

            prediction = model.predict([cleaned])[0]
            probability = model.predict_proba([cleaned])[0]

            confidence = max(probability) * 100
            risk = (1 - confidence/100) * 100

            st.markdown("---")

            # RESULT
            if prediction == 1:
                st.success("🟢 REAL NEWS")
            else:
                st.error("🔴 FAKE NEWS")

            # METRICS
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Confidence", f"{confidence:.2f}%")

            with col2:
                st.metric("Risk Score", f"{risk:.2f}%")

            with col3:
                st.metric("Word Count", len(user_input.split()))

            # PROGRESS
            st.progress(int(confidence))

            # WARNING
            if confidence < 60:
                st.warning("⚠️ Low confidence prediction. Verify manually.")

            # DETAILS
            if show_details:
                with st.expander("🔬 Detailed Analysis"):
                    st.write("### Probability Distribution")
                    st.json({
                        "Fake": float(probability[0]),
                        "Real": float(probability[1])
                    })

            # DOWNLOAD RESULT
            result_text = f"""
Prediction: {'REAL' if prediction==1 else 'FAKE'}
Confidence: {confidence:.2f}%
Risk Score: {risk:.2f}%
Time: {datetime.now()}
Text: {user_input}
"""

            st.download_button(
                "📥 Download Result",
                result_text,
                file_name="prediction.txt"
            )

        except Exception as e:
            st.error(f"Prediction error: {e}")