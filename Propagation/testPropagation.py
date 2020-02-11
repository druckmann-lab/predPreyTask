import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable 

import networkFiles as NF
from samplesPropagation import generateSamples


# General parameters that would get set in other code
layers = 10
image_size = 15
N = image_size
num_nodes = image_size**2

dtype = torch.FloatTensor

model = NF.PropagationOnly_FixedWeights(num_nodes, layers, num_nodes*5, image_size)
model.type(dtype)
testDict = generateSamples(image_size, 10000, layers)


test_dsetPath = torch.utils.data.TensorDataset(testDict["Environment"], testDict["Predator"], testDict["Range"])
loader = DataLoader(test_dsetPath, batch_size=batch, shuffle=True)



model.eval()
num_correct, num_samples = 0, 0
losses = []

# The accuracy on all pixels and path pixels can be calculated from the image labels
# Also record the loss
for env, pred, label in loader:
	# Cast the image data to the correct type and wrap it in a Variable. At
	# test-time when we do not need to compute gradients, marking the Variable
	# as volatile can reduce memory usage and slightly improve speed.
	env = Variable(env.type(dtype), requires_grad=False)
	pred = Variable(pred.type(dtype), requires_grad=False)
	label = Variable(label.type(dtype), requires_grad=False)

	# Run the model forward and compare with ground truth.
	output = model(env, pred, dtype).type(dtype)
	loss = loss_fn(output, label).type(dtype)
	preds = output.sign() 

	# Compute accuracy on ALL pixels
	num_correct += (preds.data[:, :] == label.data[:,:]).sum()
	num_samples += pred.size(0) * pred.size(1)


	losses.append(loss.data.cpu().numpy())



 

# Return the fraction of datapoints that were incorrectly classified.
accAll = 1.0 -  (float(num_correct) / (num_samples))
avg_loss = sum(losses)/float(len(losses))

print(decision)
print(label)




