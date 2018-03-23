import string
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re

# Remove all stopwords not nessary
def preprocess(sentence):
	sentence = sentence.lower()
	tokenizer = RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(sentence)
	filtered_words = filter(lambda token: token not in stopwords.words('english'), tokens)
	return list(filtered_words) 

