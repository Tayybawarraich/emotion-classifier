import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# --- NLTK Data ---
nltk.download('punkt')
nltk.download('stopwords')

# --- Load Data ---
@st.cache_data
def load_data():
    data = {
    'text': [
        # Joy
        "I am so happy and joyful today!", "This is a wonderful day.",
        "I feel cheerful and excited.", "Life feels beautiful right now.",
        "I am smiling all the time.", "Everything is going great.",
        "I feel very positive today.", "This made me really happy.",
        "I am enjoying every moment.", "I feel blessed and thankful.",
        "I love spending time with my family.", "I love my best friend so much.",
        "I feel love for everyone around me.", "I adore my partner.",
        "My heart is full of love today.", "I am grateful for my loved ones.",
        "I feel affection and joy together.", "I am so happy to see you.",
        "I cherish these wonderful moments.", "I feel emotionally connected to my friends.",
        "Today is one of the best days of my life.", "I am overjoyed!", "I feel ecstatic.", "Everything seems perfect.",
        "I am content and cheerful.", "Life is beautiful and amazing.", "I am on cloud nine.", "I am feeling jubilant.",

        # Anger
        "I feel very angry and frustrated.", "Stop shouting at me!", 
        "I am extremely annoyed right now.", "This situation makes me mad.",
        "I am furious about this.", "I cannot control my anger.",
        "This is so irritating.", "I feel rage inside me.",
        "I am upset and angry.", "Your behavior makes me angry.",
        "I hate this situation.", "I am fuming!", "This is unacceptable.", "I am enraged.",
        "I feel violent anger.", "I am mad at everything.", "I feel irritated and annoyed.", 
        "This really makes me angry.", "I cannot stand this.", "I feel agitated.",

        # Sadness
        "I feel so sad and lonely.", "I am crying because I miss you.",
        "I feel empty and broken.", "My heart feels very heavy.",
        "I am feeling depressed.", "This makes me want to cry.",
        "I feel hopeless today.", "I am emotionally drained.",
        "I feel miserable and low.", "I am very disappointed.",
        "I feel sorrowful.", "I am heartbroken.", "I feel melancholy.", 
        "Life seems gloomy.", "I am sad and down.", "I feel despair.", "I am tearful.",
        "I feel helpless.", "Nothing makes me happy today.", "I feel abandoned.",

        # Fear
        "I am terrified of the dark.", "The movie was so scary.",
        "I feel afraid and nervous.", "I am scared to go alone.",
        "This situation frightens me.", "I feel panic inside me.",
        "I am trembling with fear.", "I feel unsafe here.",
        "I am anxious about what will happen.", "That sound scared me.",
        "I feel alarmed.", "I am terrified.", "I feel horror.", "I am frightened.",
        "I feel apprehensive.", "I am feeling nervous.", "I feel dread.", "I am scared stiff.",
        "I feel shaken.", "I am in fear.",

        # Surprise
        "I am surprised by the news!", "Wow, I didn't expect that!",
        "This came as a big shock.", "I am amazed by this.",
        "I was not ready for this.", "That surprised me a lot.",
        "I am totally stunned.", "I did not see that coming.",
        "This is completely unexpected.", "I am speechless right now.",
        "I feel astonished.", "I am startled.", "I am shocked.", "I am dumbfounded.",
        "I am flabbergasted.", "I am amazed!", "I did not anticipate this.", "What a surprise!",
        "I am astounded.", "I am bewildered."
    ],

    'label': [
        # Joy
        "Joy","Joy","Joy","Joy","Joy","Joy","Joy","Joy","Joy","Joy",
        "Joy","Joy","Joy","Joy","Joy","Joy","Joy","Joy","Joy","Joy",
        "Joy","Joy","Joy","Joy","Joy","Joy","Joy","Joy",

        # Anger
        "Anger","Anger","Anger","Anger","Anger","Anger","Anger","Anger","Anger","Anger",
        "Anger","Anger","Anger","Anger","Anger","Anger","Anger","Anger","Anger","Anger",

        # Sadness
        "Sadness","Sadness","Sadness","Sadness","Sadness","Sadness","Sadness","Sadness","Sadness","Sadness",
        "Sadness","Sadness","Sadness","Sadness","Sadness","Sadness","Sadness","Sadness","Sadness","Sadness",

        # Fear
        "Fear","Fear","Fear","Fear","Fear","Fear","Fear","Fear","Fear","Fear",
        "Fear","Fear","Fear","Fear","Fear","Fear","Fear","Fear","Fear","Fear",

        # Surprise
        "Surprise","Surprise","Surprise","Surprise","Surprise","Surprise","Surprise","Surprise","Surprise","Surprise",
        "Surprise","Surprise","Surprise","Surprise","Surprise","Surprise","Surprise","Surprise","Surprise","Surprise"
    ]
}


    return pd.DataFrame(data)

# --- Preprocessing ---
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    cleaned_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(cleaned_tokens)

# --- Train Model ---
@st.cache_resource
def train_model(df):
    df['clean_text'] = df['text'].apply(preprocess_text)
    model = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', MultinomialNB()),
    ])
    model.fit(df['clean_text'], df['label'])
    return model

# --- Streamlit UI ---
st.set_page_config(page_title="Emotion Classifier", page_icon="🧠", layout="centered")

st.markdown("""
<style>
.main { background-color: #f0f4f8; }
.stButton>button { width: 100%; border-radius: 20px; background-color: #4CAF50; color: white; font-size: 16px; padding: 10px; }
.result-box { padding: 20px; border-radius: 15px; background-color: #ffffff; box-shadow: 3px 3px 8px rgba(0,0,0,0.1); text-align:center; }
.result-emoji { font-size: 80px; animation: bounce 1s infinite; }
@keyframes bounce {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-15px); }
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 NLP Emotion Classifier")
st.markdown("Type something and see which emotion it represents! Joy, Anger, Sadness, Fear, Surprise.")

# Load Data and Train Model
df = load_data()
model = train_model(df)

# User Input
st.subheader("Analyze your text")
user_input = st.text_area("What's on your mind?", placeholder="Type something like 'I had a great day!'")

if st.button("Predict Emotion"):
    if user_input.strip() == "":
        st.error("Please enter some text first.")
    else:
        cleaned_input = preprocess_text(user_input)
        prediction = model.predict([cleaned_input])[0]

        # Emoji Mapping
        emoji_dict = {
            "Joy": "😊",
            "Anger": "😡",
            "Sadness": "😢",
            "Fear": "😱",
            "Surprise": "😮"
        }
        emoji = emoji_dict.get(prediction, "🧠")

        st.markdown(f"""
            <div class="result-box">
                <div class="result-emoji">{emoji}</div>
                <h4>Detected Emotion:</h4>
                <h2>{prediction}</h2>
            </div>
        """, unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Project Info")
st.sidebar.info("Model: Multinomial Naive Bayes\nLibrary: NLTK & Scikit-learn")
if st.sidebar.checkbox("Show Training Data"):
    st.sidebar.write(df[['text', 'label']])
