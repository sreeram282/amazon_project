import streamlit as st
import pandas as pd
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Pre-load NLTK resources
nltk.data.path.append("./nltk_data")
nltk.download('punkt')

# Load data
amazon_data = pd.read_csv('amazon_product.csv')
amazon_data.drop('id', axis=1, inplace=True)

# Initialize stemmer
stemmer = SnowballStemmer('english')

# Preprocessing function
def tokenize_stem(text):
    tokens = nltk.word_tokenize(text.lower())
    stemmed = [stemmer.stem(w) for w in tokens]
    return " ".join(stemmed)

# Preprocess product data
amazon_data['stemmed_tokens'] = amazon_data.apply(
    lambda row: tokenize_stem(str(row['Title']) + " " + str(row['Description'])),
    axis=1
)

# TF-IDF setup
tfidfv = TfidfVectorizer(tokenizer=tokenize_stem)

def cosine_sim(txt1, txt2):
    matrix = tfidfv.fit_transform([txt1, txt2])
    return cosine_similarity(matrix)[0][1]

def search_product(query):
    stemmed_query = tokenize_stem(query)
    amazon_data['similarity'] = amazon_data['stemmed_tokens'].apply(lambda x: cosine_sim(stemmed_query, x))
    result = amazon_data.sort_values(by='similarity', ascending=False).head(10)[['Title', 'Description', 'Category']]
    return result

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Amazon Product Search", layout="centered")
st.title("🔍 Amazon Product Search")

product_list = amazon_data['Title'].dropna().unique().tolist()

option = st.selectbox("Choose a product from the list", [""] + product_list)

text_query = st.text_input("Or enter a product description")

if st.button("Search"):
    query = text_query if text_query else option
    if not query:
        st.warning("Please enter or select a product to search.")
    else:
        results = search_product(query)
        st.success("Top 10 matching results:")
        for index, row in results.iterrows():
            st.markdown(f"""
            **🛒 {row['Title']}**  
            *{row['Category']}*  
            {row['Description']}  
            ---
            """)