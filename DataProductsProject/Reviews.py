import pandas
import glob
import numpy as np
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim

restaurants = ["Frenchys.csv" , "Tonys.csv", "BurgerPark.csv", "Dot.csv", "Demeris.csv", "Doyles.csv", "Molinas.csv", "Cleburne.csv", "BBQInn.csv", "Brennans.csv", "Lankford.csv", "Christies.csv", "Yale.csv", "MytiBurger.csv", "ElPatio.csv", "Gaidos.csv",
"Spanish.csv", "Houstons.csv", "Pizzitolas.csv", "Brenners.csv", "Avalon.csv"]

for name in restaurants:
    data = pandas.read_csv(name, header=0)
    data['Review Date'] = pandas.to_datetime(data['Review Date'])
    print(name)

    FE1_Prior = data[(data['Review Date'] <= '2008-09-13') & (data['Review Date'] > '2008-06-13')]
    FE1_Prior_Reviews = list(FE1_Prior.Review)
    FE1_Post = data[(data['Review Date'] <= '2008-12-13') & (data['Review Date'] > '2008-09-13')]
    FE1_Post_Reviews = list(FE1_Prior.Review)

    FE2_Prior = data[(data['Review Date'] <= '2015-05-25') & (data['Review Date'] > '2015-02-25')]
    FE2_Prior_Reviews = list(FE2_Prior.Review)
    FE2_Post = data[(data['Review Date'] <= '2015-08-25') & (data['Review Date'] > '2015-05-25')]
    FE2_Post_Reviews = list(FE2_Post.Review)

    FE3_Prior = data[(data['Review Date'] <= '2016-04-18') & (data['Review Date'] > '2016-01-18')]
    FE3_Prior_Reviews = list(FE3_Prior.Review)
    FE3_Post = data[(data['Review Date'] <= '2016-07-18') & (data['Review Date'] > '2016-04-18')]
    FE3_Post_Reviews = list(FE3_Post.Review)

    FE4_Prior = data[(data['Review Date'] <= '2016-08-14') & (data['Review Date'] > '2016-05-14')]
    FE4_Prior_Reviews = list(FE4_Prior.Review)
    FE4_Post = data[(data['Review Date'] <= '2016-11-26') & (data['Review Date'] > '2016-08-14')]
    FE4_Post_Reviews = list(FE4_Post.Review)

    FE5_Prior = data[(data['Review Date'] <= '2017-08-17') & (data['Review Date'] > '2017-05-17')]
    FE5_Prior_Reviews = list(FE5_Prior.Review)
    FE5_Post = data[(data['Review Date'] <= '2017-11-27') & (data['Review Date'] > '2016-08-17')]
    FE5_Post_Reviews = list(FE5_Post.Review)

    MasterReviewList = [FE1_Prior_Reviews, FE1_Post_Reviews, FE2_Prior_Reviews, FE2_Post_Reviews, FE3_Prior_Reviews, FE3_Post_Reviews, FE4_Prior_Reviews, FE4_Post_Reviews, FE5_Prior_Reviews, FE5_Post_Reviews]
    MasterReviewListString = ["FE1_Prior_Reviews", "FE1_Post_Reviews", "FE2_Prior_Reviews", "FE2_Post_Reviews", "FE3_Prior_Reviews", "FE3_Post_Reviews", "FE4_Prior_Reviews", "FE4_Post_Reviews", "FE5_Prior_Reviews", "FE5_Post_Reviews"]
    tokenizer = RegexpTokenizer(r'\w+')
    Review_Tokenized = []
    labelCounter = 0

    for idx,reviewlist in enumerate(MasterReviewList):
        for i in range(len(reviewlist)):
            if type(reviewlist[i]) is str:
                raw = reviewlist[i].lower()
                raw = raw.replace("'", "")
                tokens = tokenizer.tokenize(raw)
                Review_Tokenized.append(tokens)

        en_stop = get_stop_words('en')
        testList = en_stop
        testList = testList.extend(['us', 'really'])

        Review_Stopped = []
        for i in range(len(Review_Tokenized)):
            stopped_tokens = [i for i in Review_Tokenized[i] if not i in en_stop]
            Review_Stopped.append(stopped_tokens)

        p_stemmer = PorterStemmer()
        Review_Stemmed = []
        for i in range(len(Review_Stopped)):
            texts = [p_stemmer.stem(i) for i in Review_Stopped[i]]
            Review_Stemmed.append(texts)

        dictionary = corpora.Dictionary(Review_Stopped)
        corpus = [dictionary.doc2bow(text) for text in Review_Stopped]

        print(MasterReviewListString[labelCounter])
        labelCounter += 1

        if len(dictionary) > 0:
            ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=2, id2word = dictionary, passes=20)
            print(ldamodel.print_topics(num_topics=2, num_words=4))
