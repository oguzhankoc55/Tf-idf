import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
# from gensim import utils
# from gensim.models.doc2vec import LabeledSentence
# from gensim.models import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os

nltk.download('stopwords')
stop_words =  set(stopwords.words('english'))
stemmer = nltk.porter.PorterStemmer()
def computeTFIDF(tfBagOfWords,idfs):
    tfidf={}
    for word,val in tfBagOfWords.items():
        tfidf[word]=val*idfs[word]
        return tfidf
def computeIDF(documents):
    import math
    N=len(documents)

    idfDict=dict.fromkeys(documents[0].keys(),0)
    for document in documents:
        for word,val in document.items():
            if val>0:
                idfDict[word]+=1
    
    for word,val in idfDict.items():
        idfDict[word]=math.log(N/float(val))
    return idfDict
def computeTF(wordDict,bagOfWords):
    tfDict={}
    bagOfWordCount=len(bagOfWords)
    for word,count in wordDict.items():
        tfDict[word]=count / float(bagOfWordCount)
    return tfDict
def review_to_wordlist(review, remove_stopwords=True):
    # Clean the text, with the option to remove stopwords.
    
    # Convert words to lower case and split them
    words = review.lower().split()

    # Optionally remove stop words (true by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    
    review_text = " ".join(words)

    # Clean the text
    review_text = re.sub(r"[^A-Za-z0-9(),!.?\'\`]", " ", review_text)
    review_text = re.sub(r"\'s", " 's ", review_text)
    review_text = re.sub(r"\'ve", " 've ", review_text)
    review_text = re.sub(r"n\'t", " 't ", review_text)
    review_text = re.sub(r"\'re", " 're ", review_text)
    review_text = re.sub(r"\'d", " 'd ", review_text)
    review_text = re.sub(r"\'ll", " 'll ", review_text)
    review_text = re.sub(r",", " ", review_text)
    review_text = re.sub(r"\.", " ", review_text)
    review_text = re.sub(r"!", " ", review_text)
    review_text = re.sub(r"\(", " ( ", review_text)
    review_text = re.sub(r"\)", " ) ", review_text)
    review_text = re.sub(r"\?", " ", review_text)
    review_text = re.sub(r"\s{2,}", " ", review_text)
    
    words = review_text.split()
    
    # Shorten words to their stems
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in words]
    
    review_text = " ".join(stemmed_words)
    
    # Return a list of words
    return(review_text)

listpath_manu= os.getcwd()+"/Dataset/reviewers"
pdfler = os.listdir(listpath_manu)
# print(pdfler[0])
i=0

list=[]
list1=[]
for pdf in pdfler:

    path_manu= os.getcwd()+"/Dataset/manuscripts/0902.1601.txt"
    path_revi= os.getcwd()+"/Dataset/reviewers/"+pdfler[i]
    
    #metin = review_to_wordlist(path_revi,True)
    metin = open(path_revi, "r")
    metin=metin.read()
    metin=metin.replace("{fenge}","")


    metin = review_to_wordlist(metin,True)
    metin1 = open(path_manu, "r")
    metin1=metin1.read()
    #print(metin1)
    bagOfWordsA=metin.split(' ')
    bagOfWordsB=metin1.split(' ')

    uniquewords=set(bagOfWordsA).union(set(bagOfWordsB))
    numOfWordsA=dict.fromkeys(uniquewords,0)
    for word in bagOfWordsA:
        numOfWordsA[word]+=1
    numOfWordsB=dict.fromkeys(uniquewords,0)
    for word in bagOfWordsB:
        numOfWordsB[word]+=1

    from nltk.corpus import stopwords
    stopwords.words('english')



    tfA=computeTF(numOfWordsA,bagOfWordsA)
    tfB=computeTF(numOfWordsB,bagOfWordsB)



    idfs=computeIDF([numOfWordsA,numOfWordsB])


    tfidfA=computeTFIDF(tfA,idfs)
    tfidfB=computeTFIDF(tfB,idfs)
    df=pd.DataFrame([tfidfA,tfidfB])

    vectorizer=TfidfVectorizer()
    vectors=vectorizer.fit_transform([metin,metin1])
    deger = ((vectors * vectors.T).A)[0,1]
    
    
    #print(path_revi)
    #print(type(deger))
    
    list.append(deger)
    list1.append(path_revi)
    i=i+1
    
# list.sort(reverse=True)

# print(list)

# feature_names=vectorizer.get_feature_names()
# dense=vectors.todense()
# denselist=dense.tolist()
# df=pd.DataFrame(denselist,columns=feature_names)
# print(df)


#print(metin1.read())
# def make_data(path):
#   veriler = []
#   pdfler = os.listdir(path)
#   for pdf in pdfler:
#     try:
#         veri= {
#             "isim":"",
#         }
#         veri.update({"isim" : pdf})
#         dosya = open(path+pdf,"r")
#         for metin in dosya:
#             veri.update({"metin":metin})
        
#         dosya.close()
#         veriler.append(veri)
#     except:
#         print( "hata")
#   return veriler

# manuscripts_veriler = make_data(path_manu)
# revievers_veriler = make_data(path_revi)
# print(manuscripts_veriler[0]["metin"])
# #print(revievers_veriler)
# # metin=[]
# metin = manuscripts_veriler[0]["metin"]


# metin = review_to_wordlist(metin,True)
# print(metin)

# for i in range(len(manuscripts_veriler)):
#   metin = manuscripts_veriler[i]["metin"]
#   metin = review_to_wordlist(metin,True)
#   manuscripts_veriler[i].update({"metin":metin})

# for i in range(len(revievers_veriler)):
#   metin = revievers_veriler[i]["metin"]
#   metin = review_to_wordlist(metin,True)
#   revievers_veriler[i].update({"metin":metin})

# vectorizer = TfidfVectorizer()

# def cosine_sim(text1, reviwers):
#     score = []
#     for i in range(len(reviwers)):
#       dic = {
#           "isim":"",
#           "result":0
#       }
#       text2 = reviwers[i]["metin"]
#       tfidf = vectorizer.fit_transform([text1, text2])
#       deger = ((tfidf * tfidf.T).A)[0,1]
#       dic.update({"isim":reviwers[i]["isim"]})
#       dic.update({"result":deger})
#       score.append(dic)
#     return score
# len(manuscripts_veriler)
# Tfidf_scores_deneme = []
# for i in range(1):
#     veri={
#         "isim":"",
#         "karsilastirma_sonucları":[]
#     }
#     score = cosine_sim(metin, revievers_veriler)
#     print(i , " ", metin)
#     veri.update({"isim":metin})
#     veri.update({"karsilastirma_sonucları":score})
#     Tfidf_scores_deneme.append(veri)

# makale = Tfidf_scores_deneme[0]["isim"]
# karsilastirma_sonucları = Tfidf_scores_deneme[0]["karsilastirma_sonucları"]
# db = pd.DataFrame(karsilastirma_sonucları)
# db = db.sort_values('result',ascending=False ,ignore_index = True )
# print(makale)
# print(db.head(10))
# path_groundtruth = os.getcwd()+"/Dataset/manuscripts/groundturth.txt"
# dosya = open(path_groundtruth,"r")
# # sonuc = ""
# # for metin in dosya:
# #   sonuc = metin
# # dosya.close()
# # groundtruth = metin
# # dizi = groundtruth.split("'")
# #print("1++++++++++++++++++",dizi[0])
# #print("2++++++++++++++++++",dizi[1])

