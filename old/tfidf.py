import codecs
from sklearn.feature_extraction.text import TfidfVectorizer

# データの用意
corpus = codecs.open('corpus.txt', 'r', 'utf-8').read().splitlines()

vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(corpus)