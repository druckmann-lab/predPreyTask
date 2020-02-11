import numpy as np
import torch
import scipy.io as sio

from generate_environment import environment
import argparse

#### Samples for training propagation with changing source pixels
# Note: There are two version

def generateSamples(N, numTraining, steps):


	# Parameters for environment generation
	N2 = N**2
	p = 0.15

	# Set up the tensors to store the
	trainEnv = np.zeros((numTraining, N2))
	trainPred = np.zeros((numTraining, N2))
	trainPredRange = np.zeros((numTraining, N2))


	runMax = np.ceil(numTraining*0.55)
	hideMax = np.ceil(numTraining*0.55)



	j = 0

	while(j < numTraining):
		X, prey, predator, cave = environment(N, p)

		# Label is 0 if cave is closer to prey (also equivalent to non-accessible to predator)
		# Label is 1 if cave is closer to predator OR non-accessible to either



		# These form the two different stopping conditions
		# Diff tracks when we've filled the whole space
		# Stop tracks when one of the ranges includes the cave
		# Label get set initially to 1, the proper value if the cave is in neither range

		q = 0

		while(q < steps):

			# Propagate Prey
			preyRange_Old = preyRange[:,:]
			del row_prey[:]
			del col_prey[:]
			row_prey = rowNext_prey[:]
			col_prey = colNext_prey[:]
			l = len(row_prey)
			del rowNext_prey[:]
			del colNext_prey[:]
			for i in range(0, l):
				row_current = row_prey[i]
				col_current = col_prey[i]
				if ((row_current != 0) and (X[(row_current - 1), col_current] == 0) and (preyRange[(row_current - 1), col_current] == 0)):
					preyRange[row_current - 1, col_current] = 1
					rowNext_prey.append(row_current - 1)
					colNext_prey.append(col_current)
				if ((row_current != N-1) and (X[row_current + 1, col_current] == 0) and (preyRange[(row_current + 1), col_current] == 0)):
					preyRange[row_current + 1, col_current] = 1
					rowNext_prey.append(row_current + 1)
					colNext_prey.append(col_current)
				if ((col_current != 0) and (X[row_current, col_current-1] == 0) and (preyRange[row_current, col_current-1] == 0)):
					preyRange[row_current, col_current-1] = 1
					rowNext_prey.append(row_current)
					colNext_prey.append(col_current-1)
				if ((col_current != N-1) and (X[row_current, col_current+1] == 0) and (preyRange[row_current, col_current+1] == 0)):
					preyRange[row_current, col_current+1] = 1
					rowNext_prey.append(row_current)
					colNext_prey.append(col_current+1)


			# Propagate predator
			predatorRange_Old = predatorRange[:,:]
			del row_predator[:]
			del col_predator[:]
			row_predator = rowNext_predator[:]
			col_predator = colNext_predator[:]
			l = len(row_predator)
			del rowNext_predator[:]
			del colNext_predator[:]
			for i in range(0, l):
				row_current = row_predator[i]
				col_current = col_predator[i]
				if ((row_current != 0) and (X[(row_current - 1), col_current] == 0) and (predatorRange[(row_current - 1), col_current] == 0)):
					predatorRange[row_current - 1, col_current] = 1
					rowNext_predator.append(row_current - 1)
					colNext_predator.append(col_current)
				if ((row_current != N-1) and (X[row_current + 1, col_current] == 0) and (predatorRange[(row_current + 1), col_current] == 0)):
					predatorRange[row_current + 1, col_current] = 1
					rowNext_predator.append(row_current + 1)
					colNext_predator.append(col_current)
				if ((col_current != 0) and (X[row_current, col_current-1] == 0) and (predatorRange[row_current, col_current-1] == 0)):
					predatorRange[row_current, col_current-1] = 1
					rowNext_predator.append(row_current)
					colNext_predator.append(col_current-1)
				if ((col_current != N-1) and (X[row_current, col_current+1] == 0) and (predatorRange[row_current, col_current+1] == 0)):
					predatorRange[row_current, col_current+1] = 1
					rowNext_predator.append(row_current)
					colNext_predator.append(col_current+1)

			q = q+1

		#print(q)

		# Change everything to +1/-1
		Xvec = np.reshape(X, (1, N2))
		Xvec = 1 - Xvec
		Xvec[Xvec == 0] = -1
		predVec = np.reshape(predator, (1, N2))
		predVec[predVec == 0] = -1
		preyVec = np.reshape(prey, (1, N2))
		preyVec[preyVec == 0] = -1


		trainEnv[j, :] = Xvec
		trainPred[j, :] = predVec
		trainPredRange[j, :] = predRangeVec



		j = j+1
		# if (j % 1000 == 0):
		# 	print(j)

	# Once all the samples are generated, return dictionary of samples

	trainEnv = torch.from_numpy(trainEnv)
	trainPred = torch.from_numpy(trainPred)
	trainPrey = torch.from_numpy(trainPrey)
	trainCave = torch.from_numpy(trainCave)
	trainLabel = torch.from_numpy(trainLabel)

	sampleDict = {"Environment": trainEnv, "Predator": trainPred, "Range": trainPredRange}

	return sampleDict


	# # Look at the output
	# fig1, ax = plt.subplots(2, 3)
	# ax[0, 0].imshow(X, cmap='Greys',  interpolation='none')
	# ax[0, 1].imshow(prey, cmap='Greys',  interpolation='none')
	# ax[0, 2].imshow(predator, cmap='Greys',  interpolation='none')
	# ax[1, 0].imshow(cave, cmap='Greys',  interpolation='none')
	# ax[1, 1].imshow(preyRange, cmap='Greys',  interpolation='none')
	# ax[1, 2].imshow(predatorRange, cmap='Greys',  interpolation='none')

	# print(label)

	# plt.show()

	# fig1.savefig('sample_label.pdf', bbox_inches = 'tight')

	
		










