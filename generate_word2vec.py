from gensim.models import word2vec

data = word2vec.Text8Corpus("alice-wakati.txt")
word_vec = word2vec.Word2Vec(data, size=100)

word_vec.save("alice-word2vec.model")

