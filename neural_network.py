# CREATE AND TRAIN THE NEURAL NETWORK


# Imports
from mesures import *
import csv
import keras
import h5py
from sklearn.cross_validation import StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np

def load_data(data_filepath, labels_filepath):
	'''Load the data from .csv files
	'''
	data = np.loadtxt(open(data_filepath, "rb"), delimiter=",")
	labels = np.loadtxt(open(labels_filepath, "rb"))
	return data, labels

def create_model():
	'''Create the neural network model
	'''
	# Build the shape of the network
	model = Sequential()
	model.add(Dense(300, input_dim=600, activation='sigmoid'))
	model.add(Dropout(0.5))
	model.add(Dense(300, activation='sigmoid'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))
	return model

def train_and_evaluate_model(model, data_train, labels_train, data_test, labels_test,):
	'''Train and evaluate the model
	'''

	filepath="weights.best.hdf5"
	
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	# Early Stopping
	early_stop=keras.callbacks.EarlyStopping(monitor='val_acc', patience=2, verbose=0, mode='auto')	
	callbacks_list = [checkpoint, early_stop]
	# Fit the model
	model.fit(data_train, labels_train, epochs=50, batch_size=5,
		  validation_data=(data_test, labels_test), callbacks=callbacks_list)
	# Test the model
	score = model.evaluate(data_test , labels_test, batch_size =5)
	
	return score

def load_trained_model(weights_path):
	''' Load an already trained model from .hdf5 files 
	'''
	model = create_model()
	model.load_weights(weights_path)
	return model

if __name__ == "__main__":
        
        model = None # Clearing the NN.
        model = load_trained_model("weights.best.hdf5")
	      
        model.compile(optimizer='rmsprop',
                        loss='binary_crossentropy',
                        metrics=['accuracy',precision,recall])
	# Prediction
        pred = np.loadtxt(open("data/N_vect_data_test3_2.csv", "rb"), delimiter=',')
        label_pred = model.predict(pred, batch_size=128)
    
        # To prevent an error 
        K.clear_session()

