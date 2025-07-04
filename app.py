import streamlit as st
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizerFast
import tensorflow as tf
import joblib
import time


@st.cache_resource
def load_model_and_tokenizer():
    model = TFDistilBertForSequenceClassification.from_pretrained("./classifier")
    tokenizer = DistilBertTokenizerFast.from_pretrained("./classifier")
    return model, tokenizer


@st.cache_resource
def load_label_encoder():
    return joblib.load("encoder.pkl")


st.set_page_config(
    page_title="NeuralNews", page_icon=":newspaper:", initial_sidebar_state="expanded"
)


model, tokenizer = load_model_and_tokenizer()
label_encoder = load_label_encoder()


st.markdown(
    """
<style>

    .main-header {
        text-align: center;
        padding: 40px 0 30px 0;
        margin-bottom: 30px;
    }
    
    .app-title {
        font-size: 3rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        text-align: center;
    }

    .app-subheader {
        font-size: 2rem;
        font-weight: 300;
        margin-bottom: 12px;
        letter-spacing: -0.02em;
        text-align: center;
        font-style: italic;
    }
    
    .app-subtitle {
        font-size: 1.1rem;
        color: #666666;
        font-weight: 400;
        text-align: center;
        margin: 0 auto;
        line-height: 1.5;
    }  
    
    .stTextInput > div > div > input {
        font-size: 1.1rem;
        padding: 18px 24px;
        transition: all 0.3s ease;
        max-width: 900px;
    }

    .stTextInput {
        height: 60px;
    }

    .stTextInput > div > div > input:focus {
        outline: none !important;
    }

    .stTextInput > div > div > input::placeholder {
        font-style: italic;
    }

    
    .stButton > button {
        border: none;
        border-radius: 12px;
        padding: 16px 32px;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        min-height: 52px;
        width: 180px;
        letter-spacing: 0.02em;
        margin-top: 16px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
    }
    
    .stButton > button:active {
        transform: translateY(0px);
    }
    
    .result-card {
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 30px;
        margin: 20px auto 0 auto;
        max-width: 500px;
        text-align: center;
        animation: slideUp 0.6s ease-out;
    }
    
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .result-label {
        font-size: 0.9rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 16px;
    }
    
    .result-category {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 12px;
        letter-spacing: -0.02em;
    }
    
    .result-headline {
        font-size: 1rem;
        font-style: italic;
        line-height: 1.5;
    }

    .result-confidence {
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 12px;
        color: #bb5a38;
    }
    
    .alert-warning {
        border-radius: 8px;
        padding: 16px;
        color: #bb5a38;
        margin: 20px auto;
        max-width: 600px;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    
    .loading-container {
        text-align: center;
        padding: 40px;
    }
    
    .loading-spinner {
        width: 40px;
        height: 40px;
        border: 3px solid #ecebe3;
        border-top: 3px solid #bb5a38;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin: 0 auto 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .loading-text {
        font-size: 1rem;
    }
    
    .app-footer {
        text-align: center;
        padding: 40px 0;
        margin-top: 40px;
        color: #666666;
        font-size: 0.9rem;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .app-title {
            font-size: 2.5rem;
            text-align: center;
        }
        
        .result-category {
            font-size: 2rem;
        }
        
        .input-section {
            padding: 0 10px;
        }
    }
    @media (max-width: 480px) {
        .app-title {
            font-size: 2rem;
            text-align: center;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)

st.sidebar.header("Dataset Details")

st.sidebar.markdown(
    """ 
    The dataset has been taken from [Kaggle](https://www.kaggle.com/datasets/rmisra/news-category-dataset). It contains around 210,000 news headlines from 2012 to 2022 from [HuffPost](https://www.huffpost.com/).

**Content** :

- `category`: Category in which the article was published.
- `headline`: Headline of the article.
- `authors`: Authors of the article.
- `link`: Link to the original article.
- `short_description`: Short description of the article.
- `date`: Date when the article was published.

There are a total of 42 categories.
"""
)

st.sidebar.subheader("Citations")

st.sidebar.markdown(
    """
1. Misra, Rishabh. "News Category Dataset." arXiv preprint arXiv:2209.11429 (2022).
2. Misra, Rishabh and Jigyasa Grover. "Sculpting Data for ML: The first act of Machine Learning." ISBN 9798585463570 (2021).
"""
)

st.sidebar.markdown("[Dataset Source](https://rishabhmisra.github.io/publications)")

st.markdown(
    """
<div class="main-header">
    <h1 class="app-title">NeuralNews</h1>
    <h5 class="app-subheader">~Classifying the world, one headline at a time</h5>
    <p class="app-subtitle">News classification using fine-tuned DistilBERT. Enter a headline, get its category.</p>
</div>
""",
    unsafe_allow_html=True,
)

headline = st.text_area(
    "Headline",
    placeholder="e.g. NASA finds new planet",
    key="headline_input",
)


predict_clicked = st.button("Classify", key="predict_btn", type="primary")


if predict_clicked:
    if not headline.strip():
        st.markdown(
            """
        <div class="alert-warning">
            Please enter a headline to classify.
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        loading_placeholder = st.empty()
        with loading_placeholder.container():
            st.markdown(
                """
            <div class="loading-container">
                <div class="loading-spinner"></div>
                <div class="loading-text">Classifying headline...</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        time.sleep(5)
        loading_placeholder.empty()

        inputs = tokenizer(
            headline,
            return_tensors="tf",
            truncation=True,
            padding="max_length",
            max_length=128,
        )
        logits = model(inputs).logits
        pred = tf.argmax(logits, axis=1).numpy()[0]
        category = label_encoder.inverse_transform([pred])[0]
        confidence_scores = tf.nn.softmax(logits, axis=1).numpy()[0]
        confidence = confidence_scores[pred] 


        st.markdown(
            f"""
    <div class="result-card">
        <div class="result-label">Category</div>
        <div class="result-category">{category}</div>
        <div class="result-headline">"{headline}"</div>
        <div class="result-confidence">Confidence: {confidence * 100:.2f}%</div>
    </div>
    """,
            unsafe_allow_html=True,
        )


st.markdown(
    """
<div class="app-footer">
    Built with <a href="https://streamlit.io">Streamlit</a> • Powered by <a href="https://huggingface.co/docs/transformers/index">Transformers</a>
</div>
""",
    unsafe_allow_html=True,
)
