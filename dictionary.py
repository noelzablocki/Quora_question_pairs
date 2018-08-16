import numpy as np

def makeDictionary(vector_filepath, word_filepath):
	''' Each word is associated with his vector representation
	'''
	# Load all the vect
	embedding = np.load(vector_filepath)
	# print(embedding.shape)

	# Load the corresponding words
	with open(word_filepath,"r") as file:
		vocab = file.read().splitlines()

	# Create the dictionary
	embedding_dict = {word: embedding[i] for i, word in enumerate(vocab)}
	return embedding_dict


if __name__ == '__main__':
	vector_filepath = "models/noel_embeddings.npy"
	word_filepath = "models/wiki_vocab.txt"
	embedding_dict = makeDictionary(vector_filepath, word_filepath)
