import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import collections
from torch.autograd import Variable
from weightMask import generateFixedWeightMask_PredPrey

#### These are the top level network files for the decision network on the task

class FixedPropagation_PredPrey(torch.nn.Module):
	def __init__(self, num_pixels, layers, H_decision, imageSize):
		super(FixedPropagation_PredPrey, self).__init__()
		weightMask, diagMask, edgeMask, cornerMask = generateFixedWeightMask_PredPrey(imageSize)
		self.iteratedLayer = RecurrentDecision_FixedPropagation(num_pixels, num_pixels, layers, weightMask, diagMask, edgeMask, cornerMask)
		

	def forward(self, X, prey, pred, cave, dtype):	
		
		prey_range, pred_range, decision = self.iteratedLayer(prey, pred, X, X, cave, dtype)
		
		# This block is to fix a problem when only one element is passed in
		# The first dimension is normally the batch, but the repeated layers will squeeze out this dimension
		# Add it back in before concatenating

		# if (pred_range.dim() == 1):
		# 	pred_range.unsqueeze(0)
		
		return prey_range, pred_range, decision


class FixedDecision_PredPrey(torch.nn.Module):
	def __init__(self, num_pixels, layers, H_decision, imageSize):
		super(FixedDecision_PredPrey, self).__init__()
		weightMask, diagMask, edgeMask, cornerMask = generateFixedWeightMask_PredPrey(imageSize)
		self.iteratedLayer = RecurrentDecision_FixedDecision(num_pixels, num_pixels, layers, weightMask, diagMask, edgeMask, cornerMask)
		

	def forward(self, X, prey, pred, cave, dtype):	
		
		prey_range, pred_range, decision = self.iteratedLayer(prey, pred, X, X, cave, dtype)
		
		# This block is to fix a problem when only one element is passed in
		# The first dimension is normally the batch, but the repeated layers will squeeze out this dimension
		# Add it back in before concatenating

		# if (pred_range.dim() == 1):
		# 	pred_range.unsqueeze(0)
		
		return prey_range, pred_range, decision


#### These are sub-functions for the decision task

class RecurrentDecision_FixedDecision(torch.nn.Module):
	def __init__(self, D_input, hidden, layers, weightMask, diagMask, edgeMask, cornerMask):
		super(RecurrentDecision_FixedDecision, self).__init__()

		# We have two propagation layers to set up
		# 1 will be the prey
		# 2 will be the predator

		self.mask = weightMask
		self.diagMask = diagMask
		self.invertMask = ~self.mask #torch.ones((hidden, hidden)).type(torch.ByteTensor) - self.mask
		self.invertDiag = ~self.diagMask #torch.ones((hidden, hidden)).type(torch.ByteTensor) - self.diagMask
		self.iteration = layers
		self.tanh = nn.Tanh()
		self.sigmoid = nn.Sigmoid()



		# These are the prey parameters
		# For this case we want these to remain fixed
		self.hiddenWeight1 = nn.Parameter(torch.ones(hidden, hidden), requires_grad=False)
		self.bias1 = nn.Parameter(torch.randn(hidden, 1), requires_grad=False)
		self.inputWeight1 = nn.Parameter(torch.ones(D_input, hidden), requires_grad=False)
		self.scalar1 = nn.Parameter(torch.ones(1)*20, requires_grad=False)

		# Set up the hidden weights
		self.w1 = nn.Parameter(torch.randn(hidden, 1), requires_grad=True)
		# self.w1.data[:] = 0.25
		# self.w1.data[edgeMask] = 0.34
		# self.w1.data[cornerMask] = 0.5
		self.hiddenWeight1.data[self.invertMask] = 0
		self.hiddenWeight1.data[self.mask] = 1


		# Set up the input weights
		self.a1 = nn.Parameter(torch.ones(hidden, 1), requires_grad=False)
		self.inputWeight1.data[:] = 0
		self.inputWeight1.data[self.diagMask] = 1
		self.bias1.data[:] = -0.15





		# These are the predator parameters
		self.hiddenWeight2 = nn.Parameter(torch.ones(hidden, hidden), requires_grad=False)
		self.bias2 = nn.Parameter(torch.randn(hidden, 1), requires_grad=False)
		self.inputWeight2 = nn.Parameter(torch.ones(D_input, hidden), requires_grad=False)
		self.scalar2 = nn.Parameter(torch.ones(1)*20, requires_grad=False)

		# Set up the hidden weights
		self.w2 = nn.Parameter(torch.randn(hidden, 1), requires_grad=True)
		# self.w2.data[:] = 0.25
		# self.w2.data[edgeMask] = 0.34
		# self.w2.data[cornerMask] = 0.5
		self.hiddenWeight2.data[self.invertMask] = 0
		self.hiddenWeight2.data[self.mask] = 1


		# Set up the input weights
		self.a2 = nn.Parameter(torch.ones(hidden, 1), requires_grad=False)
		self.inputWeight2.data[:] = 0
		self.inputWeight2.data[self.diagMask] = 1
		self.bias2.data[:] = -0.15

		

	def forward(self, initial_hidden1, initial_hidden2, input1, input2, cave, dtype):
		u1 = initial_hidden1.clone()
		u1_fix = initial_hidden1.clone()
		u1_fix[u1_fix==-1]=0
		u1_fix_expand = u1_fix.unsqueeze(-1)
		input1_expand = input1.unsqueeze(-1)


		u2 = initial_hidden2.clone()
		u2_fix = initial_hidden2.clone()
		u2_fix[u2_fix==-1]=0
		u2_fix_expand = u2_fix.unsqueeze(-1)
		input2_expand = input2.unsqueeze(-1)

		batch = list(u1.size())
		batch = batch[0]
		

		decision = torch.zeros(batch, 2).type(dtype)
		cave[cave==-1] = 0
		# idx = torch.nonzero(cave)

		

		for _ in range(0, self.iteration):
			u1_expand = u1.unsqueeze(-1)
			v1_expand = u1_fix_expand + torch.matmul((self.w1.expand_as(self.hiddenWeight1) * self.hiddenWeight1), u1_expand) + (self.bias1) + \
				torch.matmul((self.a1.expand_as(self.inputWeight1) * self.inputWeight1), input1_expand)
			v1 = v1_expand.squeeze()
			u1 = self.tanh(v1 * self.scalar1.expand_as(v1))
			#u = torch.sign(u)


			u2_expand = u2.unsqueeze(-1)
			v2_expand = u2_fix_expand + torch.matmul((self.w2.expand_as(self.hiddenWeight2) * self.hiddenWeight2), u2_expand) + (self.bias2) + \
				torch.matmul((self.a2.expand_as(self.inputWeight2) * self.inputWeight2), input2_expand)
			v2 = v2_expand.squeeze()
			u2 = self.tanh(v2 * self.scalar2.expand_as(v2))


			decision[:, 0] = self.sigmoid((-100.0*decision[:, 1].clone() + 20*torch.sum(cave*u2, dim=1)))
			decision[:, 1] = self.sigmoid((-100.0*decision[:, 0].clone() + 20*torch.sum(cave*u1, dim=1)))




			# print(torch.sum(cave*u1))
			# print(u2[idx])
			# print(u1[idx])
			# print(decision)

		return u1, u2, decision


class RecurrentDecision_FixedPropagation(torch.nn.Module):
	def __init__(self, D_input, hidden, layers, weightMask, diagMask, edgeMask, cornerMask):
		super(RecurrentDecision_FixedPropagation, self).__init__()

		# We have two propagation layers to set up
		# 1 will be the prey
		# 2 will be the predator

		self.mask = weightMask
		self.diagMask = diagMask
		self.invertMask = ~self.mask #torch.ones((hidden, hidden)).type(torch.ByteTensor) - self.mask
		self.invertDiag = ~self.diagMask #torch.ones((hidden, hidden)).type(torch.ByteTensor) - self.diagMask
		self.iteration = layers
		self.tanh = nn.Tanh()
		self.sigmoid = nn.Sigmoid()



		# These are the prey parameters
		# For this case we want these to remain fixed
		self.hiddenWeight1 = nn.Parameter(torch.ones(hidden, hidden), requires_grad=False)
		self.bias1 = nn.Parameter(torch.ones(hidden, 1), requires_grad=False)
		self.inputWeight1 = nn.Parameter(torch.ones(D_input, hidden), requires_grad=False)
		self.scalar1 = nn.Parameter(torch.ones(1)*20, requires_grad=False)

		# Set up the hidden weights
		self.w1 = nn.Parameter(torch.ones(hidden, 1), requires_grad=False)
		self.w1.data[:] = 0.25
		self.w1.data[edgeMask] = 0.34
		self.w1.data[cornerMask] = 0.5
		self.hiddenWeight1.data[self.invertMask] = 0
		self.hiddenWeight1.data[self.mask] = 1


		# Set up the input weights
		self.a1 = nn.Parameter(torch.ones(hidden, 1), requires_grad=False)
		self.inputWeight1.data[:] = 0
		self.inputWeight1.data[self.diagMask] = 1
		self.bias1.data[:] = -0.15





		# These are the predator parameters
		self.hiddenWeight2 = nn.Parameter(torch.ones(hidden, hidden), requires_grad=False)
		self.bias2 = nn.Parameter(torch.ones(hidden, 1), requires_grad=False)
		self.inputWeight2 = nn.Parameter(torch.ones(D_input, hidden), requires_grad=False)
		self.scalar2 = nn.Parameter(torch.ones(1)*20, requires_grad=False)

		# Set up the hidden weights
		self.w2 = nn.Parameter(torch.ones(hidden, 1), requires_grad=False)
		self.w2.data[:] = 0.25
		self.w2.data[edgeMask] = 0.34
		self.w2.data[cornerMask] = 0.5
		self.hiddenWeight2.data[self.invertMask] = 0
		self.hiddenWeight2.data[self.mask] = 1


		# Set up the input weights
		self.a2 = nn.Parameter(torch.ones(hidden, 1), requires_grad=False)
		self.inputWeight2.data[:] = 0
		self.inputWeight2.data[self.diagMask] = 1
		self.bias2.data[:] = -0.15

		# These are the decision variables
		self.w_decision = nn.Parameter(5*torch.randn(2, 1), requires_grad=True)
		self.winhibit_decision = nn.Parameter(-100*torch.randn(2, 1), requires_grad=False)

		

	def forward(self, initial_hidden1, initial_hidden2, input1, input2, cave, dtype):
		u1 = initial_hidden1.clone()
		u1_fix = initial_hidden1.clone()
		u1_fix[u1_fix==-1]=0
		u1_fix_expand = u1_fix.unsqueeze(-1)
		input1_expand = input1.unsqueeze(-1)


		u2 = initial_hidden2.clone()
		u2_fix = initial_hidden2.clone()
		u2_fix[u2_fix==-1]=0
		u2_fix_expand = u2_fix.unsqueeze(-1)
		input2_expand = input2.unsqueeze(-1)

		batch = list(u1.size())
		batch = batch[0]
		

		decision = torch.zeros(batch, 2).type(dtype)
		cave[cave==-1] = 0
		idx = torch.nonzero(cave)

		

		for _ in range(0, self.iteration):
			u1_expand = u1.unsqueeze(-1)
			v1_expand = u1_fix_expand + torch.matmul((self.w1.expand_as(self.hiddenWeight1) * self.hiddenWeight1), u1_expand) + (self.bias1) + \
				torch.matmul((self.a1.expand_as(self.inputWeight1) * self.inputWeight1), input1_expand)
			v1 = v1_expand.squeeze()
			u1 = self.tanh(v1 * self.scalar1.expand_as(v1))
			#u = torch.sign(u)


			u2_expand = u2.unsqueeze(-1)
			v2_expand = u2_fix_expand + torch.matmul((self.w2.expand_as(self.hiddenWeight2) * self.hiddenWeight2), u2_expand) + (self.bias2) + \
				torch.matmul((self.a2.expand_as(self.inputWeight2) * self.inputWeight2), input2_expand)
			v2 = v2_expand.squeeze()
			u2 = self.tanh(v2 * self.scalar2.expand_as(v2))


			decision[:, 0] = self.sigmoid((self.winhibit_decision[1]*decision[:, 1].clone() + self.w_decision[0]*torch.sum(cave*u2, dim=1)))
			decision[:, 1] = self.sigmoid((self.winhibit_decision[0]*decision[:, 0].clone() + self.w_decision[1]*torch.sum(cave*u1, dim=1)))




			#print(torch.sum(cave*u1))
			# print(u2[idx])
			# print(u1[idx])
			# print(decision)

		return u1, u2, decision


# This is the correct way to do fixed weights

		# self.hiddenWeight1 = nn.Parameter(torch.ones(hidden, hidden), requires_grad=False)
		# self.bias1 = nn.Parameter(torch.ones(hidden, 1), requires_grad=False)
		# self.inputWeight1 = nn.Parameter(torch.ones(D_input, hidden), requires_grad=False)
		# self.scalar1 = nn.Parameter(torch.ones(1)*20, requires_grad=False)

		# # Set up the hidden weights
		# self.w1 = nn.Parameter(torch.ones(hidden, 1), requires_grad=True)
		# self.w1.data[:] = 0.25
		# self.w1.data[edgeMask] = 0.34
		# self.w1.data[cornerMask] = 0.5
		# self.hiddenWeight1.data[self.invertMask] = 0
		# self.hiddenWeight1.data[self.mask] = 1


		# # Set up the input weights
		# self.a1 = nn.Parameter(torch.ones(hidden, 1), requires_grad=False)
		# self.inputWeight1.data[:] = 0
		# self.inputWeight1.data[self.diagMask] = 1
		# self.bias1.data[:] = -0.15





		# # These are the predator parameters
		# self.hiddenWeight2 = nn.Parameter(torch.ones(hidden, hidden), requires_grad=False)
		# self.bias2 = nn.Parameter(torch.ones(hidden, 1), requires_grad=False)
		# self.inputWeight2 = nn.Parameter(torch.ones(D_input, hidden), requires_grad=False)
		# self.scalar2 = nn.Parameter(torch.ones(1)*20, requires_grad=False)

		# # Set up the hidden weights
		# self.w2 = nn.Parameter(torch.ones(hidden, 1), requires_grad=False)
		# self.w2.data[:] = 0.25
		# self.w2.data[edgeMask] = 0.34
		# self.w2.data[cornerMask] = 0.5
		# self.hiddenWeight2.data[self.invertMask] = 0
		# self.hiddenWeight2.data[self.mask] = 1


		# # Set up the input weights
		# self.a2 = nn.Parameter(torch.ones(hidden, 1), requires_grad=False)
		# self.inputWeight2.data[:] = 0
		# self.inputWeight2.data[self.diagMask] = 1
		# self.bias2.data[:] = -0.15







