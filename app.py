import streamlit as st
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import os
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(page_title="Speech Intelligence Dashboard", layout="wide")

# ==========================
# NLTK DOWNLOAD
# ==========================
try:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True) # Add this line
    nltk.download("vader_lexicon", quiet=True)
    nltk.download("stopwords", quiet=True)
except Exception as e:
    st.error(f"Error downloading NLTK data: {e}")

sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words("english"))

# ==========================
# SESSION STATE
# ==========================
if "text_input" not in st.session_state:
    st.session_state.text_input = ""

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================
# HEADER
# ==========================
st.title("🚀 Speech Intelligence & NLP Dashboard")
st.markdown(
    "Analyze speeches using **Sentiment Analysis, Topic Modeling, and Text Intelligence**."
)

st.markdown("---")

# ==========================
# MODERN HERO SECTION (CLEAN CARD STYLE)
# ==========================

st.markdown("""
<style>
.feature-card {
    background-color: #111827;
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 15px;
    border: 1px solid #1f2937;
}
.feature-title {
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 6px;
}
.feature-desc {
    font-size: 14px;
    color: #cbd5e1;
}
</style>
""", unsafe_allow_html=True)

colA, colB = st.columns([2, 1])

with colA:
    st.markdown("## 📌 What This Dashboard Does")

    st.markdown("""
    <div class="feature-card">
        <div class="feature-title">🔍 Sentence-Level Sentiment Analysis</div>
        <div class="feature-desc">
        Analyzes each sentence individually and assigns a sentiment score 
        (Positive, Neutral, or Negative).
        </div>
    </div>

    <div class="feature-card">
        <div class="feature-title">📊 Sentiment Distribution</div>
        <div class="feature-desc">
        Displays how sentiment is distributed across the entire speech 
        using aggregated counts.
        </div>
    </div>

    <div class="feature-card">
        <div class="feature-title">📈 Sentiment Trend Visualization</div>
        <div class="feature-desc">
        Shows how emotional tone changes from the beginning 
        to the end of the speech.
        </div>
    </div>

    <div class="feature-card">
        <div class="feature-title">☁️ Word Cloud Generation</div>
        <div class="feature-desc">
        Visualizes the most frequently used words — larger words 
        appear more often in the speech.
        </div>
    </div>

    <div class="feature-card">
        <div class="feature-title">🔗 Bigram Extraction</div>
        <div class="feature-desc">
        Identifies the most frequent two-word combinations 
        to reveal recurring phrases and themes.
        </div>
    </div>

    <div class="feature-card">
        <div class="feature-title">🧠 Topic Modeling (LDA)</div>
        <div class="feature-desc">
        Uses Latent Dirichlet Allocation to uncover hidden thematic 
        topics within the speech.
        </div>
    </div>
    """, unsafe_allow_html=True)

with colB:
    st.markdown("""
    <div style="
        background-color:#1e293b;
        padding:25px;
        border-radius:15px;
        text-align:center;
        margin-top:55px;
        border:1px solid #334155;
    ">
        <h4>👉 How to Use</h4>
        <p style="color:#cbd5e1;">
        Select input from the sidebar <br>
        then click <b>Analyze</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ==========================
# SIDEBAR INPUT
# ==========================
st.sidebar.header("Select Input Source")

option = st.sidebar.radio(
    "Choose Input Method:",
    ("Select Predefined Speech", "Upload Your Own File", "Enter Custom Text")
)

if option == "Select Predefined Speech":

    speeches = {
        "Bose Speech": os.path.join(BASE_DIR, "Sentiment_Analysis_Bose.txt"),
        "Gandhiji Speech": os.path.join(BASE_DIR, "Sentiment_Analysis_Gandhi.txt"),
        "Mandela Speech": os.path.join(BASE_DIR, "Sentiment_Analysis_Mandela.txt")
    }

    selected_speech = st.sidebar.selectbox("Select a speech:", list(speeches.keys()))

    if st.sidebar.button("Load Speech"):
        file_path = speeches[selected_speech]
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                st.session_state.text_input = f.read()
            st.success(f"{selected_speech} loaded successfully!")
        else:
            st.error("Speech file not found!")

elif option == "Upload Your Own File":
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded_file is not None:
        st.session_state.text_input = uploaded_file.read().decode("utf-8")
        st.success("File uploaded successfully!")

elif option == "Enter Custom Text":
    st.session_state.text_input = st.text_area(
        "Enter your text here:",
        value=st.session_state.text_input,
        height=200
    )

# ==========================
# EMPTY STATE
# ==========================
if st.session_state.text_input.strip() == "":
    st.markdown("""
    <div style='text-align:center; padding:50px;'>
        <h3>🚀 Ready to Analyze a Speech?</h3>
        <p>Select a speech or upload text from the sidebar.</p>
    </div>
    """, unsafe_allow_html=True)

# ==========================
# TEXT PREVIEW + QUICK METRICS
# ==========================
if st.session_state.text_input.strip() != "":
    text_data = st.session_state.text_input

    st.subheader("📄 Text Preview")
    st.write(text_data[:800] + "...")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Words", len(word_tokenize(text_data)))
    col2.metric("Total Characters", len(text_data))
    col3.metric("Estimated Reading Time", f"{len(text_data.split())//200 + 1} min")

    st.caption("""
    • Total Words: Number of words in the speech.  
    • Total Characters: Total length including spaces.  
    • Estimated Reading Time: Approximate reading time assuming 200 words per minute.
    """)

    st.markdown("---")

# ==========================
# NLP FUNCTIONS
# ==========================
def analyze_sentiment(text):
    sentences = sent_tokenize(text)
    scores = [sia.polarity_scores(sentence)["compound"] for sentence in sentences]
    return sentences, scores


def extract_bigrams(text, top_n=10):
    tokens = word_tokenize(text.lower())
    tokens = [w for w in tokens if w.isalpha() and w not in stop_words]
    bigram_list = list(ngrams(tokens, 2))
    return Counter(bigram_list).most_common(top_n)


def extract_topics(text, n_topics=3):
    documents = [doc.strip() for doc in text.split("\n\n") if doc.strip()]
    if len(documents) < 2:
        return ["Not enough text for topic modeling."]

    vectorizer = CountVectorizer(stop_words="english")
    dtm = vectorizer.fit_transform(documents)

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(dtm)

    words = vectorizer.get_feature_names_out()
    topics = []

    for idx, topic in enumerate(lda.components_):
        top_words = [words[i] for i in topic.argsort()[:-6:-1]]
        topics.append(f"Topic {idx+1}: " + ", ".join(top_words))

    return topics

# ==========================
# ANALYZE BUTTON
# ==========================
if st.button("Analyze"):

    if st.session_state.text_input.strip() == "":
        st.warning("Please provide text first.")
    else:
        text_data = st.session_state.text_input
        sentences, scores = analyze_sentiment(text_data)
        avg_score = sum(scores) / len(scores)

        tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Sentiment Details", "🧠 Advanced NLP"])

        # TAB 1
        with tab1:
            st.subheader("📊 Overall Sentiment Summary")

            col1, col2, col3 = st.columns(3)
            col1.metric("Average Sentiment", f"{avg_score:.3f}")
            col2.metric("Total Sentences", len(sentences))
            col3.metric("Unique Words", len(set(word_tokenize(text_data))))

            st.caption("""
            Average Sentiment shows overall emotional tone (-1 negative to +1 positive).
            """)

            result_df = pd.DataFrame({
                "Sentence": sentences,
                "Compound Score": scores
            })

            def label(score):
                if score >= 0.05:
                    return "Positive"
                elif score <= -0.05:
                    return "Negative"
                return "Neutral"

            result_df["Sentiment"] = result_df["Compound Score"].apply(label)

            st.subheader("📊 Sentiment Distribution")
            st.bar_chart(result_df["Sentiment"].value_counts())

            st.caption("Displays count of Positive, Neutral, and Negative sentences.")

        # TAB 2
        with tab2:
            st.subheader("🔍 Sentence-Level Sentiment")
            st.dataframe(result_df)

            st.caption("Each sentence is scored individually using VADER sentiment analysis.")

            st.subheader("📈 Sentiment Trend")
            fig, ax = plt.subplots()
            ax.plot(scores)
            ax.set_xlabel("Sentence Index")
            ax.set_ylabel("Compound Score")
            st.pyplot(fig)

            st.caption("Shows how sentiment evolves throughout the speech.")

        # TAB 3
        with tab3:
            st.subheader("☁️ Word Cloud")
            wordcloud = WordCloud(width=800, height=400, background_color="black").generate(text_data)
            fig_wc, ax_wc = plt.subplots()
            ax_wc.imshow(wordcloud, interpolation="bilinear")
            ax_wc.axis("off")
            st.pyplot(fig_wc)

            st.caption("Frequently used words appear larger in the cloud.")

            st.subheader("🔗 Top Bigrams")
            bigrams = extract_bigrams(text_data)
            bigram_df = pd.DataFrame(bigrams, columns=["Bigram", "Frequency"])
            bigram_df["Bigram"] = bigram_df["Bigram"].apply(lambda x: " ".join(x))
            st.dataframe(bigram_df)

            st.caption("Shows most common two-word combinations.")

            st.subheader("🧠 Extracted Topics (LDA)")
            topics = extract_topics(text_data)
            for topic in topics:
                st.write(topic)

            st.caption("Topic modeling uncovers hidden thematic clusters in the speech.")

