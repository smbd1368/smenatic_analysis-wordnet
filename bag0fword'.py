import json
import time
from gevent import os
from nltk.corpus import wordnet
import pandas as pd
import matplotlib.pyplot as plt
from langdetect.detector import Detector
from nltk.translate import AlignedSent, Alignment
from nltk.corpus import comtrans
from langdetect import detect
import nltk
import pycountry
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
import string
import re
import chardet
# nltk.download()
# nltk.download('punkt')
# nltk.download('stopwords')
# set(stopwords.words('english'))
from textblob import TextBlob
from nltk.corpus import wordnet as wn
import goslate
from langdetect import detect

from googletrans import Translator
translator = Translator()




ALL_data = " "


from nltk.corpus import wordnet

# print(dog)



def bag_of_words(words):
    return dict([(word, True) for word in words])


wordfreq = {}
stop = 1
all_data = list()
for r, d, f in os.walk('/home/mammad/PycharmProjects/textmining/Hw1/dataset/politifact23'):
    if len(f) >= 1:
        # print(f[0])
        ifile = open(str(r) + "/" + str(f[0]))
        for i, line in enumerate(ifile):

            if i % 10000 == 0:
                print()
            if i == stop:
                break

            data = json.loads(line)
            text = data['text']
            title = data['title']
            data_null = str(text) + " " + str(title)

            # data_null = data_null.replace("  ", "")

            # data_null = data_null[0]

            # print(data_null,"_______________________________+++++++++=")
            try:
                trans = translator.translate(str(data_null), dest='en')
                data_null = trans.text
                # print(data_null)
            except:
                print()




            ALL_data =  data_null + " " + ALL_data



            # lang = detect(data_null)
            # print(lang)
            # gs = goslate.Goslate()
            # print(gs.translate(str(data_null), str(lang)))



            # b = TextBlob(data_null)
            # print(b.detect_language())
            # print(chardet.detect(data_null))

            # print(data_null)

            # lan=chardet.detect(data_null.encode('cp1251'))
            #
            # print(lan['language'],"sssss")
            # print(data_null)



            # print(detect_language(data_null))
            # print()
            # if lan=="":
            #     lan="English"


            # if lan['language'] != "" and lan['language'] != "None":
            #     print(str(lan['language']).translate(data_null),"aaaasadasdasdadadadadasda")






        # corpus = nltk.sent_tokenize(ALL_data)
        text_token = nltk.wordpunct_tokenize(ALL_data)
        ll = [x for x in text_token if not re.fullmatch('[' + string.punctuation + ']+', x)]
        corpus =ll
        stemmer = nltk.PorterStemmer()
        singles = [stemmer.stem(plural) for plural in corpus]



        # print(singles)
        # print(corpus)


        # print(corpus)
        # print(singles)

        # frequency word in bag of word after tokenenize and punc and stemmer
        for i in range(len(singles)):
            corpus[i] = corpus[i].lower()
            corpus[i] = re.sub(r'\W', ' ', corpus[i])
            corpus[i] = re.sub(r'\s+', ' ', corpus[i])

        wordfreq = {}
        for sentence in corpus:
            tokens = nltk.word_tokenize(sentence)
            for token in tokens:
                # print(token)
                if token not in wordfreq.keys():
                    # print(detect("War doesn't show who's right, just who's left."),"aaaaaaaaaaaaaaaaaa")

                    # algnsent = AlignedSent(token)
                    # print(algnsent.words)
                    # print(algnsent.mots,'aaaaaaa')
                    wordfreq[token] = 1
                else:
                    wordfreq[token] += 1
            # print(wordfreq)

print(wordfreq)  # print(ll)
for x in wordfreq:
    # print(x)
    syns = wordnet.synsets(str(x))

    if len(syns)>=1:
        # print(syns[0].name())
        print(x," :  ",syns[0].lemmas()[0].name())

        # wordnet.synset(x).lowest_common_hypernyms(wordnet.synset(x))




for x in wordfreq:
    for ss in wn.synsets('green'):
        print (ss, ss.hypernyms())
        # print(wn.synset(ss))


# print(wordnet.synsets('friend')[0].name())
for x in wordfreq:
    for y in wordfreq:
        synonynm = wordnet.synsets(x)
        synonynp = wordnet.synsets(y)
        # wordnet.synsets('friend')[0].examples()
        if len(synonynm)>=1 and len(synonynp)>=1:
            a=synonynm[0].name()
            b=synonynp[0].name()
            wordnet.synset(a)
            a1=wordnet.synset(a)
            b1 = wordnet.synset(b)
            print(a,"'",b," :  " )
            print(a1.path_similarity(b1),"path")  #comparisons are the path similarity
            # print(a1.wup_similarity(b1),"hypernym")  #Synsets occur relative to each other in the hypernym tree


        # print(x.shortest_path_distance(x))









