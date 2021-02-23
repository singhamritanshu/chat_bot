import nltk 
import numpy as np

nltk.download('punkt') # This is a package with a pre trainned tokenizer.
from nltk.stem.porter import PorterStemmer 
stemmer = PorterStemmer()
def token(sentence):
    return nltk.word_tokenize(sentence)

def steaming(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenised_sent,all_words):
    '''
    sentence=["hello","how","are","you"]
    words=["hi","hello","I","you","bye","thank","cool"]
    bag=[0,1,0,1,0,0,0]
    '''
    tokenised_sent = [steaming(i) for i in tokenised_sent]
    # Creating an array of same size 
    bag = np.zeros(len(all_words),dtype=np.float32)
    for idx,w in enumerate(all_words):
        if w in tokenised_sent:
            bag[idx] = 1.0
    return bag
#sentence=["hello","how","are","you"]
#words=["hi","hello","I","you","bye","thank","cool"]
#bag = bag_of_words(sentence,words)
#print(bag)