import nltk
import requests
import re
import numpy as np
import pandas as pd
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def parse_text(review):
    tokens = word_tokenize(review)
    tokens = [w.lower() for w in tokens]

    stripped = [re.sub("[^\w\s]", "", w) for w in tokens]

    words = [w for w in stripped if w.isalpha()]

    words = [w for w in words if not w in set(stopwords.words('english'))]
      
    lemmatizer = WordNetLemmatizer() 
    
    words = np.array([lemmatizer.lemmatize(w) for w in words])

    return list(np.unique(words))
        
