from flask import Flask, request, render_template
import pandas as pd
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('punkt')

# Flask app
amazon_recommendation = Flask(__name__)

# Load data and prepare
amazon_data = pd.read_csv('amazon_product.csv')
amazon_data.drop('id', axis=1, inplace=True)

stemmer = SnowballStemmer('english')

def tokenize_stem(text):
    tokens = nltk.word_tokenize(text.lower())
    stemmed = [stemmer.stem(w) for w in tokens]
    return " ".join(stemmed)

amazon_data['stemmed_tokens'] = amazon_data.apply(
    lambda row: tokenize_stem(str(row['Title']) + " " + str(row['Description'])),
    axis=1
)

tfidfv = TfidfVectorizer(tokenizer=tokenize_stem)

def cosine_sim(txt1, txt2):
    matrix = tfidfv.fit_transform([txt1, txt2])
    return cosine_similarity(matrix)[0][1]

def search_product(query):
    stemmed_query = tokenize_stem(query)
    amazon_data['similarity'] = amazon_data['stemmed_tokens'].apply(lambda x: cosine_sim(stemmed_query, x))
    result = amazon_data.sort_values(by='similarity', ascending=False).head(10)[['Title', 'Description', 'Category']]
    return result

@amazon_recommendation.route('/', methods=['GET', 'POST'])
def index():
    results = None
    if request.method == 'POST':
        query = request.form.get('query')
        if query:
            results = search_product(query).to_dict(orient='records')
    return render_template('amazon_recom.html', results=results)

if __name__ == '__main__':
    amazon_recommendation.run(debug=True)
