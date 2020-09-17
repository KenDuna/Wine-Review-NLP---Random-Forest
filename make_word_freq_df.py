import nltk
import requests
import re
import pandas as pd
from bs4 import BeautifulSoup
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

data = pd.read_csv('X_train.csv')

reviews = data['description']
final_list = []

for review in reviews:
    
    tokens = word_tokenize(review)
    tokens = [w.lower() for w in tokens]

    stripped = [re.sub("[^\w\s]", "", w) for w in tokens]

    words = [w for w in stripped if w.isalpha()]

    words = [w for w in words if not w in set(stopwords.words('english'))]
      
    lemmatizer = WordNetLemmatizer() 
    
    words = [lemmatizer.lemmatize(w) for w in words]

    for w in words:
        final_list.append(w)
        
fdist = FreqDist(final_list)

df = pd.DataFrame.from_dict(fdist, orient='index')
df.columns = ['Frequency']
df.index.name = 'Word'
df = df.sort_values(by='Frequency', ascending = False)
df.to_csv('word_freq.csv')

fdist.plot(30, cumulative=False)

