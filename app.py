import streamlit as st
import joblib


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)


# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("fake_news_svm_pipeline.pkl")


model = load_model()


# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666666;
        margin-bottom: 30px;
    }

    .info-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #f5f5f5;
        margin-bottom: 20px;
    }

    .result-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 25px;
        font-weight: 700;
        margin-top: 20px;
    }

    .fake-box {
        background-color: #ffe6e6;
        color: #c62828;
    }

    .real-box {
        background-color: #e6f7e6;
        color: #2e7d32;
    }

    .footer {
        text-align: center;
        color: #777777;
        font-size: 14px;
        margin-top: 40px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    '<div class="main-title">📰 Fake News Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Detect whether a news article is likely to be Fake or Real using NLP and Machine Learning.'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# Information Box
# --------------------------------------------------

st.info(
    "💡 Enter the news headline and article content below, "
    "then click **Check News** to get the model prediction."
)


# --------------------------------------------------
# Input Section
# --------------------------------------------------

st.subheader("📝 Enter News Details")

title = st.text_input(
    "News Title",
    placeholder="Example: Government announces a new economic policy"
)

text = st.text_area(
    "News Content",
    placeholder="Paste the complete news article here...",
    height=250
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🔍 Check News", use_container_width=True):

    if not title.strip():
        st.warning("⚠️ Please enter a news title.")

    elif not text.strip():
        st.warning("⚠️ Please enter the news content.")

    else:

        # Combine title and article
        combined_text = title + " " + text

        # Prediction
        prediction = model.predict([combined_text])[0]

        # Decision score
        score = model.decision_function([combined_text])[0]


        # --------------------------------------------------
        # Result
        # --------------------------------------------------

        if prediction == 0:

            st.markdown(
                '<div class="result-box fake-box">'
                '🔴 FAKE NEWS'
                '</div>',
                unsafe_allow_html=True
            )

            st.write(
                "The model classified this article as **Fake News**."
            )

        else:

            st.markdown(
                '<div class="result-box real-box">'
                '🟢 REAL NEWS'
                '</div>',
                unsafe_allow_html=True
            )

            st.write(
                "The model classified this article as **Real News**."
            )


        # --------------------------------------------------
        # Decision Score
        # --------------------------------------------------

        st.subheader("📊 Model Information")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Decision Score",
                f"{score:.4f}"
            )

        with col2:

            if score < 0:
                st.metric(
                    "Predicted Class",
                    "Fake"
                )
            else:
                st.metric(
                    "Predicted Class",
                    "Real"
                )


        # --------------------------------------------------
        # Score Explanation
        # --------------------------------------------------

        with st.expander("ℹ️ What is Decision Score?"):

            st.write(
                "The decision score indicates how strongly the SVM model "
                "leans toward one of the two classes."
            )

            st.write(
                "**Negative score → Fake News**"
            )

            st.write(
                "**Positive score → Real News**"
            )

            st.caption(
                "Note: Decision score is not a probability."
            )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown(
    '<div class="footer">'
    'Built using Python • NLP • TF-IDF • Support Vector Machine'
    '</div>',
    unsafe_allow_html=True
)