# Imports
import numpy as np

def makeDictionary():
	# Load all the vect
	embedding = np.load("models/noel_embeddings.npy")
	# print(embedding.shape)

	# Load the corresponding words
	with open("models/wiki_vocab.txt","r") as f:
		vocab = f.read().splitlines()

	# print(len(vocab))
	# print(vocab[0])

	# Create the dictionary
	embedding_dic = {word: embedding[i] for i, word in enumerate(vocab)}

	# embedding_dic["the"]
	return embedding_dic
