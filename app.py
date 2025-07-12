import nltk 

nltk.download('punkt') # punkt is a sentence and word tokenizer 

nltk.download('stopwords') # stopwords are common words that do not add much meaning to a sentence. for example, "the", "is", "in", etc.

nltk.download('wordnet') # WordNet is a lexical database for the English language that helps in lemmatization(getting the root word). 

nltk.download('omw-1.4') # Open Multilingual Wordnet, used for lemmatization in multiple languages

nltk.download("averaged_perceptron_tagger") # Part-of-speech tagging, which helps in understanding the role of a word in a sentence

nltk.download('averaged_perceptron_tagger_eng') # English part-of-speech tagging

import streamlit as st 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
from nltk.corpus.reader.wordnet import NOUN, VERB, ADJ, ADV 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return ADJ 
    elif tag.startswith('V'):
        return VERB 
    elif tag.startswith('N'):
        return NOUN 
    elif tag.startswith('R'):
        return ADV
    else:
        return NOUN

def preprocess(text):
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english')) 
    filtererd_words = [word for word in words if word.lower() not in stop_words]
    pos_tags = nltk.pos_tag(filtererd_words)
    lemmatizer = WordNetLemmatizer()
    lemmatized = [
        lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)) 
        for word, tag in pos_tags   
    ]

    return " ".join(lemmatized)

def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text)
    compound = score['compound'] 

    if compound >= 0.5 :
        return "Positive 😊"
    elif compound <= -0.5 :
        return "Negative 😞" 
    else :
        return "Neutral 😐" 


st.title("🧠 NLP Sentiment Analyzer")
st.write("Enter a sentence to check the mood!")

user_input = st.text_area("✍️ Type your sentence here:")

if st.button("Analyze Sentiment"):
    cleaned = preprocess(user_input)
    result = analyze_sentiment(cleaned)

    st.subheader("🧽 Cleaned Text")
    st.write(cleaned)

    st.subheader("🔍 Sentiment Result")
    st.success(result)




