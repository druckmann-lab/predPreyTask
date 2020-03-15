import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable 
from torch.utils.data import DataLoader
import matplotlib as mpl
from matplotlib.colors import colorConverter

import networkFiles as NF
from samplesDecision import generateSamples


# General parameters that would get set in other code
layers = 2
image_size = 20
N = image_size
num_nodes = image_size**2

dtype = torch.FloatTensor

model = NF.FixedPropagation_PredPrey(num_nodes, layers, num_nodes*5, image_size)
model.type(dtype)
testDict = generateSamples(image_size, 2, layers)


test_dsetPath = torch.utils.data.TensorDataset(testDict["Environment"], testDict["Predator"], testDict["Prey"], testDict["Cave"], testDict["Label"])
loader = DataLoader(test_dsetPath, batch_size=32, shuffle=True)

loss_fn = nn.BCELoss()

model.eval()
num_correct, num_samples = 0, 0
losses = []

# The accuracy on all pixels and path pixels can be calculated from the image labels
# Also record the loss
for env, pred, prey, cave, label in loader:
	# Cast the image data to the correct type and wrap it in a Variable. At
	# test-time when we do not need to compute gradients, marking the Variable
	# as volatile can reduce memory usage and slightly improve speed.
	env = Variable(env.type(dtype), requires_grad=False)
	pred = Variable(pred.type(dtype), requires_grad=False)
	prey = Variable(prey.type(dtype), requires_grad=False)
	cave = Variable(cave.type(dtype), requires_grad=False)
	label = Variable(label.type(dtype), requires_grad=False)

	# Run the model forward and compare with ground truth.
	prey_range, pred_range, output = model(env, prey, pred, cave, dtype)
	loss = loss_fn(output, label).type(dtype)
	#preds = output.sign() 

	# Compute accuracy on ALL pixels
	num_correct += (torch.argmax(label, dim = 1) == torch.argmax(output, dim = 1)).sum()
	num_samples += label.size(0)


	losses.append(loss.data.cpu().numpy())

print(output)
print(label)
print(loss)

errorAll = 1.0 -  (float(num_correct) / (num_samples))
print(errorAll)
print(num_samples)

 

cmap1 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap',['white','black'],256)
cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2',['white','skyblue'],256)
cmap3 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2',['white','blue'],256)
cmap4 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2',['white','pink'],256)

cmap2._init() # create the _lut array, with rgba values
cmap3._init()
cmap4._init()

# create your alpha array and fill the colormap with them.
# here it is progressive, but you can create whathever you want
alphas = np.linspace(0, 0.8, cmap2.N+3)
cmap2._lut[:,-1] = alphas
cmap3._lut[:,-1] = alphas
cmap4._lut[:,-1] = alphas


# # Look at the output
fig1, ax = plt.subplots(1, 2)

#ax[0, 0].imshow(np.reshape(output[0,:].detach().numpy(), (N, N)), cmap='Greys',  interpolation='none')
ax[0].imshow(-1*np.reshape(env[0,:].numpy(), (N, N)), interpolation='none', cmap=cmap1, origin='lower')
ax[0].imshow(np.reshape(pred_range[0,:].detach().numpy(), (N, N)), interpolation='none', cmap=cmap2, origin='lower')
ax[0].imshow(np.reshape(pred[0,:].detach().numpy(), (N, N)), interpolation='none', cmap=cmap3, origin='lower')
ax[1].imshow(-1*np.reshape(env[0,:].numpy(), (N, N)), interpolation='none', cmap=cmap1, origin='lower')
ax[1].imshow(np.reshape(prey_range[0,:].detach().numpy(), (N, N)), interpolation='none', cmap=cmap4, origin='lower')
ax[1].imshow(np.reshape(prey[0,:].detach().numpy(), (N, N)), interpolation='none', cmap=cmap3, origin='lower')


plt.show()
