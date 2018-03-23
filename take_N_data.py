# READ THE TRAINNING SET
# AND TAKE THE 100 FIRST EXAMPLES

import csv
import itertools
import nltk
from dictionary import *
from sentPrepro import *


# Constantes
nbExamples = 400000 
i = 1
dictionary = makeDictionary()

# Open a file to read
with open('kaggel_data/train.csv') as csvfileR:
	spamreader = csv.reader(csvfileR)
	
	# Open a file to write
	with open('N_vect_data.csv', 'w', newline='') as csvfileW:
		spamwriter = csv.writer(csvfileW)
	
		print(' dealing with example : :')	
		for row in spamreader:
			# Turn the couples questions into vector
			# Merge them into one vect and write it on a file 
			if (i > 1) and (i != 105782) and (i != 201843):
				
				# Remove the stopwords from the sentence
				wordsQ1 = preprocess(row[3])
				vectIni = 0
				nbWords = len(wordsQ1) 
				
				# If the sentence was only made of stopword,
				# set the vector as null
				if nbWords==0:
					vectQ1 = np.zeros(300)
				else:
					for words in wordsQ1:
						try:
							temps = dictionary[words]
							if vectIni==0 :
								vectQ1 = temps
								vectIni = 1
							else:
								vectQ1 = vectQ1+temps
						except:
							pass
					vectQ1= 1/nbWords*vectQ1
	
				# Remove the stopwords from the sentence
				wordsQ2 = preprocess(row[4])
				vectIni = 0
				nbWords = len(wordsQ2)

				# If the sentence was only made of stopwords
				# set the vector as null
				if nbWords==0:
					vectQ2 = np.zeros(300)
				else:
					for words in wordsQ2:
						try:
							temps = dictionary[words]
							if vectIni==0:
								vectQ2 = temps
								vectIni = 1
							else:
								vectQ2 = vectQ2+temps
						except:
							pass
					vectQ2 = 1/nbWords*vectQ2

				# Concatenate the 2 Questions
				vect = np.concatenate((vectQ1,vectQ2))
				spamwriter.writerow(vect)		
			print(i)
			i = i + 1
			
			# End when nbExample-1 have been read
			if i > nbExamples:
				break
