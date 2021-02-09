import re, string, unicodedata
import nltk
import contractions
import inflect
import pandas as pd
from bs4 import BeautifulSoup
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

def strip_html(str):
    soup = BeautifulSoup(str, "html.parser")
    return soup.get_text()

def denoise_text(str):
    str = strip_html(str)
    str = replace_contractions(str)
    return str

def removeStopword(str):
    stop_words = set(stopwords.words('stopwords_id'))
    word_tokens = word_tokenize(str)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return ' '.join(filtered_sentence)

def replace_contractions(str):
    """Replace contractions in string of text"""
    return contractions.fix(str)

def removeNonAscii(str):
    str = unicodedata.normalize('NFKD', str).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    str = re.sub("b'|b\"",'',str)
    return str

def removeWhiteSpace(str):
    str = re.sub('[\s]+', ' ', str)
    return str

def removeURL(str):
    str = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', str)
    return str

def removePunctuation(str):
    str = re.sub(r'[^\w]|_',' ',str)
    return str

def removeDigitFromString(str):
    str = re.sub("\S*\d\S*", "", str).strip()
    return str

def removeDigit(str):
    str = re.sub(r"\b\d+\b", " ", str)
    return str

def cleaning(str):
    str = removeNonAscii(str)
    str = removeWhiteSpace(str)
    str = removeURL(str)
    str = removePunctuation(str)
    str = removeDigitFromString(str)
    str = removeDigit(str)
    str = str.lower()
    
    return str

def removeTwitterSymbols(str):
    #remove RT
    str = re.sub('RT', '', str)
    #remove @username
    str = re.sub('@[^\s]+','',str)

    return str
    
def doPreprocessing(str):
    str = removeTwitterSymbols(str)
    str = denoise_text(str)
    str = cleaning(str)
    str = removeStopword(str)
    
    return str
