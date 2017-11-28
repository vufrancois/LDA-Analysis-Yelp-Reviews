import pandas
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim

data = pandas.read_csv("BBQinn.csv", header=0)
Reviews = list(data.Review)
#text = text.replace("'", "")
tokenizer = RegexpTokenizer(r'\w+')

Review_Tokenized = []
for i in range(len(Reviews)):
    if type(Reviews[i]) is str:
        raw = Reviews[i].lower()
        tokens = tokenizer.tokenize(raw)
        Review_Tokenized.append(tokens)


en_stop = get_stop_words('en')

Review_Stopped = []
for i in range(len(Review_Tokenized)):
    stopped_tokens = [i for i in Review_Tokenized[i] if not i in en_stop]
    Review_Stopped.append(stopped_tokens)


#p_stemmer = PorterStemmer()

#Review_Stemmed = []
#for i in range(len(Review_Stopped)):
    #texts = [p_stemmer.stem(i) for i in Review_Stopped[i]]
    #Review_Stemmed.append(texts)

dictionary = corpora.Dictionary(Review_Stopped)
corpus = [dictionary.doc2bow(text) for text in Review_Stopped]


ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, id2word = dictionary, passes=20)
print(ldamodel.print_topics(num_topics=5, num_words=4))
