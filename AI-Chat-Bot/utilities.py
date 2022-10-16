import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer 
stemmer=PorterStemmer()
def tokenize(sentence):

    return nltk.word_tokenize(sentence)
def stem(word=""):
    word=word.lower()
    return stemmer.stem(word)




def bag_of_words(tokenized_sentence, words):

    sentence_words = [stem(word) for word in tokenized_sentence]
    
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag

