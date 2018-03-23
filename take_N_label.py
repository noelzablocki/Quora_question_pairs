# READ THE TRAINNING SET
# AND TAKE THE 100 FIRST EXAMPLES

import csv
import numpy as np

# Constantes
nbExamples = 400000 
i = 1

# Open a file to read
with open('kaggel_data/train.csv') as csvfileR:
	spamreader = csv.reader(csvfileR)
	
	# Open a file to write
	with open('N_vect_label.csv', 'w') as csvfileW:
		spamwriter = csv.writer(csvfileW)
		
		for row in spamreader:
			# Write label 
			if (i > 1) and (i != 105782) and (i != 201843):
				label = row[5]
				spamwriter.writerow(label)		
			i = i + 1

			# End when nbExample-1 have been read
			if i > nbExamples:
				break
