import random
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st
import plotly.express as px


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Sentiment Analysis — AI Emotion Analyzer",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# LOAD TRAINED MODEL AND FILES
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "Model"

try:
    model = joblib.load(MODEL_DIR / "svm_model2.pkl")
    vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
    emotion_labels = joblib.load(MODEL_DIR / "emotion_labels.pkl")

except FileNotFoundError as error:
    st.error(
        f"Required model file not found: {error.filename}. "
        "Place all three pickle files in the same folder as app.py."
    )
    st.stop()


# =========================================================
# SESSION STATE
# =========================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "review_text" not in st.session_state:
    st.session_state.review_text = ""

if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

if "last_generated" not in st.session_state:
    st.session_state.last_generated = None


# =========================================================
# EMOTION INFORMATION
# =========================================================

EMOTION_INFO = {
    "Sadness": {
        "icon": "😢",
        "color": "#60A5FA",
        "description": "The text expresses sadness or disappointment."
    },
    "Anger": {
        "icon": "😠",
        "color": "#F87171",
        "description": "The text expresses anger, frustration, or irritation."
    },
    "Love": {
        "icon": "❤️",
        "color": "#F472B6",
        "description": "The text expresses affection, love, or appreciation."
    },
    "Surprise": {
        "icon": "😲",
        "color": "#FBBF24",
        "description": "The text expresses surprise or unexpected excitement."
    },
    "Fear": {
        "icon": "😨",
        "color": "#A78BFA",
        "description": "The text expresses fear, worry, or nervousness."
    },
    "Joy": {
        "icon": "😊",
        "color": "#34D399",
        "description": "The text expresses happiness or enjoyment."
    }
}


# =========================================================
# DIFFERENT SAMPLE REVIEWS
# =========================================================
# These are example sentences, not assigned emotion labels.
# The trained model determines the emotion of each sentence.

SAMPLE_REVIEWS = [
    "I had an amazing day today, and everything went better than I expected.",
    "I feel lonely because my friends have not spoken to me for days.",
    "The customer service was terrible, and I am extremely frustrated.",
    "I love spending time with my family because they make me feel special.",
    "I am nervous about my interview tomorrow and cannot stop thinking about it.",
    "Wow, I never expected to receive such a wonderful birthday surprise!",
    "The movie had beautiful music, excellent acting, and an unforgettable story.",
    "I worked hard on this project, but the result was disappointing.",
    "My best friend surprised me with a thoughtful gift, and I cannot stop smiling.",
    "I am worried that I might not perform well in my upcoming examination.",
    "The meeting was cancelled at the last minute, which really annoyed me.",
    "I feel grateful for everyone who supported me during a difficult time.",
    "I was shocked when I discovered what had happened behind the scenes.",
    "I miss the happy moments I used to share with my old friends.",
    "The competition was challenging, but I enjoyed every minute of it.",
    "I am afraid of making the same mistake again.",
    "This restaurant provided excellent food and friendly service.",
    "I cannot believe how badly everything went during the presentation.",
    "Receiving that message made me feel loved and appreciated.",
    "I am thrilled because I finally completed something I have worked on for months.",
    "The weather ruined our plans, and everyone seemed disappointed.",
    "I was completely surprised when my colleagues organized a celebration for me.",
    "I feel uncomfortable speaking in front of a large audience.",
    "The unfair decision made me feel angry and ignored.",
    "Spending the evening with my favourite people made me feel peaceful and happy.",
    "I felt disappointed when the event did not meet my expectations.",
    "I am excited to explore new opportunities and learn new skills.",
    "I was frightened when I heard a strange noise outside my window.",
    "The kind words from my teacher made me feel appreciated.",
    "I never imagined that such an ordinary day would turn out to be so memorable."
]


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_emotion_name(prediction):
    """Convert the model's numeric prediction to an emotion name."""
    prediction = int(prediction)

    if isinstance(emotion_labels, dict):
        return str(emotion_labels.get(prediction, prediction)).title()

    if isinstance(emotion_labels, (list, tuple)):
        return str(emotion_labels[prediction]).title()

    return str(prediction).title()


def predict_emotion(text):
    """Transform text using the saved TF-IDF vectorizer and predict emotion."""
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]
    return get_emotion_name(prediction)


def add_to_history(text, emotion, source):
    """Save a prediction in this session's history."""
    record = {
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Sentence / Review": text,
        "Predicted Emotion": emotion,
        "Source": source
    }

    st.session_state.history.insert(0, record)


def generate_review():
    """Generate a review different from the previous generated review."""
    available_reviews = [
        review for review in SAMPLE_REVIEWS
        if review != st.session_state.last_generated
    ]

    # If the pool is exhausted, allow all reviews again.
    if not available_reviews:
        available_reviews = SAMPLE_REVIEWS.copy()

    review = random.choice(available_reviews)

    st.session_state.last_generated = review
    st.session_state.review_text = review


def analyze_current_text(source):
    """Predict the emotion and add the result to the history."""
    text = st.session_state.review_text.strip()

    if not text:
        st.session_state.message = (
            "Please generate a review or enter a sentence first."
        )
        return

    emotion = predict_emotion(text)

    add_to_history(text, emotion, source)

    st.session_state.last_result = {
        "text": text,
        "emotion": emotion,
        "time": datetime.now().strftime("%H:%M:%S")
    }

    st.session_state.message = ""


# Initialize result state.
if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "message" not in st.session_state:
    st.session_state.message = ""


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🎭 AI Emotion Analyzer")
    st.caption("Understand emotions through text.")

    st.divider()

    st.markdown("### 🎨 Appearance")

    selected_theme = st.radio(
        "Choose dashboard theme",
        ["Dark", "Light"],
        index=0 if st.session_state.theme == "Dark" else 1,
        horizontal=True,
        key="theme_selector"
    )

    st.session_state.theme = selected_theme

    st.divider()

    st.markdown("### 🧭 Navigation")

    page = st.radio(
        "Choose a section",
        [
            "🏠 Dashboard",
            "📜 Prediction History",
            "📊 Emotion Analytics",
            "ℹ️ About Project"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### 🧠 Model Information")
    st.markdown("**Algorithm:** Support Vector Machine")
    st.markdown("**Text features:** TF-IDF")
    st.markdown("**Task:** Emotion classification")
    st.markdown("**Classes:** 6 emotions")

    st.caption("Created by SHAHRUKH KHAN")


# =========================================================
# THEME STYLING
# =========================================================

if st.session_state.theme == "Dark":

    background = "#0B1120"
    surface = "#151E30"
    text_color = "#F1F5F9"
    muted_color = "#A5B4CC"
    border_color = "#29364D"

else:

    background = "#F4F7FB"
    surface = "#FFFFFF"
    text_color = "#172033"
    muted_color = "#64748B"
    border_color = "#DFE6F0"


st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {background};
        color: {text_color};
    }}

    [data-testid="stSidebar"] {{
        background-color: {surface};
        border-right: 1px solid {border_color};
    }}

    [data-testid="stHeader"] {{
        background-color: transparent;
    }}

    h1, h2, h3, h4, p, label {{
        color: {text_color};
    }}

    .hero {{
        padding: 30px;
        border-radius: 22px;
        background: linear-gradient(125deg, #312E81, #4338CA, #7C3AED);
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 12px 35px rgba(79, 70, 229, 0.20);
    }}

    .hero h1 {{
        color: white;
        font-size: 34px;
        margin-bottom: 10px;
    }}

    .hero p {{
        color: #E0E7FF;
        font-size: 15px;
    }}

    .metric-card {{
        background-color: {surface};
        border: 1px solid {border_color};
        border-radius: 16px;
        padding: 20px;
        min-height: 125px;
    }}

    .metric-title {{
        color: {muted_color};
        font-size: 13px;
        margin-bottom: 8px;
    }}

    .metric-value {{
        color: {text_color};
        font-size: 27px;
        font-weight: 700;
    }}

    .emotion-card {{
        background-color: {surface};
        border: 1px solid {border_color};
        border-radius: 16px;
        padding: 18px;
        text-align: center;
        margin-bottom: 12px;
    }}

    .result-card {{
        background-color: {surface};
        border: 1px solid {border_color};
        border-radius: 18px;
        padding: 25px;
        margin-top: 15px;
        margin-bottom: 20px;
    }}

    .result-emotion {{
        font-size: 32px;
        font-weight: 800;
        margin-top: 8px;
        margin-bottom: 10px;
    }}

    .small-muted {{
        color: {muted_color};
        font-size: 13px;
    }}

    div.stButton > button {{
        border-radius: 10px;
        min-height: 44px;
        font-weight: 600;
        border: 1px solid {border_color};
        transition: 0.2s ease;
    }}

    div.stButton > button:hover {{
        border-color: #818CF8;
        transform: translateY(-1px);
    }}

    div[data-testid="stTextArea"] textarea {{
        background-color: {surface};
        color: {text_color};
        border-radius: 12px;
        border: 1px solid {border_color};
    }}

    .footer {{
        color: {muted_color};
        text-align: center;
        padding: 20px 0 10px 0;
        font-size: 13px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <h1>🎭 Sentiment Analysis — AI Emotion Analyzer</h1>
        <p>
            Discover the emotions behind every sentence.
            Generate sample reviews, analyze your own text,
            and explore your prediction history.
        </p>
        <p><b>Created by SHAHRUKH KHAN</b></p>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# DASHBOARD PAGE
# =========================================================

if page == "🏠 Dashboard":

    total_predictions = len(st.session_state.history)

    emotion_counts = {}

    for record in st.session_state.history:
        emotion = record["Predicted Emotion"]
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

    most_common_emotion = (
        max(emotion_counts, key=emotion_counts.get)
        if emotion_counts else "—"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">TOTAL PREDICTIONS</div>
                <div class="metric-value">{total_predictions}</div>
                <div class="small-muted">Current session</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">EMOTIONS DETECTED</div>
                <div class="metric-value">{len(emotion_counts)}</div>
                <div class="small-muted">Different classes found</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">MOST FREQUENT EMOTION</div>
                <div class="metric-value">{most_common_emotion}</div>
                <div class="small-muted">Based on session history</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("## ✨ Analyze a Review or Sentence")

    st.write(
        "Generate an example review automatically or enter your own text."
    )

    left, right = st.columns([1, 1])

    with left:
        if st.button(
            "🎲 Generate Review or Sentence",
            type="primary",
            use_container_width=True
        ):
            generate_review()

    with right:
        if st.button(
            "🧹 Clear Text",
            use_container_width=True
        ):
            st.session_state.review_text = ""
            st.session_state.message = ""
            st.rerun()

    user_text = st.text_area(
        "Your review or sentence",
        key="review_text",
        height=150,
        max_chars=5000,
        placeholder=(
            "Click 'Generate Review or Sentence' or type a review here..."
        )
    )

    st.caption(
        "Tip: The generator selects a different sample from its review pool "
        "on consecutive clicks."
    )

    analyze_col, reset_col = st.columns([2, 1])

    with analyze_col:
        if st.button(
            "🚀 Predict Emotion",
            type="primary",
            use_container_width=True
        ):
            analyze_current_text("Generated" if (
                st.session_state.review_text
                and st.session_state.review_text == st.session_state.last_generated
            ) else "Manual")

    with reset_col:
        if st.button(
            "🔄 Reset Result",
            use_container_width=True
        ):
            st.session_state.last_result = None
            st.session_state.message = ""
            st.rerun()

    if st.session_state.message:
        st.warning(st.session_state.message)

    # Show latest prediction.
    result = st.session_state.last_result

    if result:

        emotion = result["emotion"]
        emotion_info = EMOTION_INFO.get(
            emotion,
            {
                "icon": "🧠",
                "color": "#818CF8",
                "description": "The model predicted this emotion."
            }
        )

        st.markdown("## 🧠 Prediction Result")

        st.markdown(
            f"""
            <div class="result-card">
                <div class="small-muted">PREDICTED EMOTION</div>
                <div class="result-emotion"
                     style="color: {emotion_info['color']};">
                    {emotion_info['icon']} {emotion}
                </div>
                <p>{emotion_info['description']}</p>
                <div class="small-muted">
                    Prediction time: {result['time']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("**Analyzed text:**")
        st.info(result["text"])

        st.caption(
            "This is the model's predicted class, not a clinical assessment "
            "or a guaranteed interpretation of a person's feelings."
        )

    st.divider()

    st.markdown("## 🌈 Explore the Six Emotions")

    emotion_names = [
        "Joy", "Sadness", "Anger",
        "Love", "Fear", "Surprise"
    ]

    columns = st.columns(3)

    for index, emotion in enumerate(emotion_names):

        info = EMOTION_INFO[emotion]
        count = emotion_counts.get(emotion, 0)

        with columns[index % 3]:
            st.markdown(
                f"""
                <div class="emotion-card">
                    <div style="font-size: 30px;">{info['icon']}</div>
                    <h3 style="color: {info['color']};">{emotion}</h3>
                    <p class="small-muted">{info['description']}</p>
                    <b>{count}</b>
                    <div class="small-muted">Predictions this session</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.divider()

    st.markdown("## 🕘 Recent Predictions")

    if st.session_state.history:

        recent_df = pd.DataFrame(
            st.session_state.history[:5]
        )

        st.dataframe(
            recent_df,
            use_container_width=True,
            hide_index=True
        )

        csv_data = pd.DataFrame(
            st.session_state.history
        ).to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Download Prediction History",
            data=csv_data,
            file_name="emotion_prediction_history.csv",
            mime="text/csv"
        )

    else:
        st.info(
            "Your prediction history will appear here after your first "
            "successful prediction."
        )


# =========================================================
# PREDICTION HISTORY PAGE
# =========================================================

elif page == "📜 Prediction History":

    st.markdown("## 📜 Prediction History")

    if st.session_state.history:

        history_df = pd.DataFrame(st.session_state.history)

        st.write(
            f"Total recorded predictions: **{len(history_df)}**"
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        csv_data = history_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Export History as CSV",
            data=csv_data,
            file_name="emotion_prediction_history.csv",
            mime="text/csv"
        )

        st.warning(
            "Clearing history removes the recorded predictions "
            "from this Streamlit session."
        )

        if st.button("🗑️ Clear All Prediction History"):

            st.session_state.history = []
            st.session_state.last_result = None
            st.success("Prediction history cleared.")
            st.rerun()

    else:
        st.info(
            "No predictions have been recorded yet. "
            "Go to the Dashboard to analyze a sentence."
        )


# =========================================================
# EMOTION ANALYTICS PAGE
# =========================================================

elif page == "📊 Emotion Analytics":

    st.markdown("## 📊 Emotion Analytics")

    if st.session_state.history:

        history_df = pd.DataFrame(st.session_state.history)

        counts_df = (
            history_df["Predicted Emotion"]
            .value_counts()
            .rename_axis("Emotion")
            .reset_index(name="Count")
        )

        color_map = {
            emotion: info["color"]
            for emotion, info in EMOTION_INFO.items()
        }

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:

            fig_bar = px.bar(
                counts_df,
                x="Emotion",
                y="Count",
                color="Emotion",
                color_discrete_map=color_map,
                title="Emotion Distribution",
                text="Count"
            )

            fig_bar.update_layout(
                showlegend=False,
                paper_bgcolor=background,
                plot_bgcolor=surface,
                font_color=text_color
            )

            st.plotly_chart(
                fig_bar,
                use_container_width=True
            )

        with chart_col2:

            fig_pie = px.pie(
                counts_df,
                names="Emotion",
                values="Count",
                color="Emotion",
                color_discrete_map=color_map,
                hole=0.58,
                title="Prediction Breakdown"
            )

            fig_pie.update_layout(
                paper_bgcolor=background,
                font_color=text_color
            )

            st.plotly_chart(
                fig_pie,
                use_container_width=True
            )

        st.markdown("### 📈 Emotion Summary")

        st.dataframe(
            counts_df,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info(
            "Analyze some sentences first to see your emotion analytics."
        )


# =========================================================
# ABOUT PAGE
# =========================================================

elif page == "ℹ️ About Project":

    st.markdown("## ℹ️ About This Project")

    st.markdown(
        """
        ### 🎭 Sentiment Analysis — AI Emotion Analyzer

        This dashboard uses a machine-learning model to classify text
        into one of six emotion categories.

        **Model pipeline**

        1. A user enters a sentence or generates a sample review.
        2. The saved TF-IDF vectorizer converts the text into numerical
           features.
        3. The trained Support Vector Machine (SVM) predicts an emotion.
        4. The result is displayed and added to the session history.

        **Supported emotion classes**

        - 😊 Joy
        - 😢 Sadness
        - 😠 Anger
        - ❤️ Love
        - 😨 Fear
        - 😲 Surprise

        **Technologies used**

        - Python
        - Streamlit
        - Pandas
        - Scikit-learn
        - TF-IDF Vectorization
        - Support Vector Machine
        - Plotly
        - Joblib

        **Important:** The model predicts a category based on patterns
        learned from its training data. It may misclassify text,
        sarcasm, context, or mixed emotions. Its prediction should not
        be treated as a definitive assessment of a person's feelings.
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        Built with Python, Streamlit & Machine Learning<br>
        Created by <b>SHAHRUKH KHAN</b>
    </div>
    """,
    unsafe_allow_html=True
)

