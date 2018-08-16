# Imports
import numpy as np

def makeDictionary():
	''' Make a dictionnary. Each word is associated with a vector
	'''
	# Load all the vect
	embedding = np.load("models/noel_embeddings.npy")
	# print(embedding.shape)

	# Load the corresponding words
	with open("models/wiki_vocab.txt","r") as f:
		vocab = f.read().splitlines()

	# Create the dictionary
	embedding_dic = {word: embedding[i] for i, word in enumerate(vocab)}
	return embedding_dic
