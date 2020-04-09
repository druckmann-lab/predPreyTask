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
layers = 8
image_size = 20
N = image_size
num_nodes = image_size**2
r = 3

dtype = torch.FloatTensor

model = NF.FixedAll_PredPrey(num_nodes, layers, num_nodes*5, image_size)
model.type(dtype)

testDict = generateSamples(image_size, r, layers)


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
	prey_range, pred_range, output, trace = model(env, prey, pred, cave, dtype)
	loss = loss_fn(output, label).type(dtype)
	#preds = output.sign() 

	# Compute accuracy on ALL pixels
	num_correct += (torch.argmax(label, dim = 1) == torch.argmax(output, dim = 1)).sum()
	num_samples += label.size(0)


	losses.append(loss.data.cpu().numpy())

# print(output)
# print(label)
# print(loss)

# errorAll = 1.0 -  (float(num_correct) / (num_samples))
# print(errorAll)
# print(num_samples)

 

cmap1 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap',['white','black'],256)
cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2',['white','skyblue'],256)
cmap3 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2',['white','blue'],256)
cmap4 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2',['white','pink'],256)
cmap5 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2',['white','mediumvioletred'],256)
cmap6 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2',['white','goldenrod'],256)

cmap2._init() # create the _lut array, with rgba values
cmap3._init()
cmap4._init()
cmap5._init()
cmap6._init()


# create your alpha array and fill the colormap with them.
# here it is progressive, but you can create whathever you want
alphas = np.linspace(0, 0.8, cmap2.N+3)
cmap2._lut[:,-1] = alphas
cmap3._lut[:,-1] = alphas
cmap4._lut[:,-1] = alphas
cmap5._lut[:,-1] = alphas
cmap6._lut[:,-1] = alphas


# # Look at the output
fig1, ax = plt.subplots(r, 3)

for q in range(r):
#ax[0, 0].imshow(np.reshape(output[0,:].detach().numpy(), (N, N)), cmap='Greys',  interpolation='none')
	ax[q, 0].imshow(-1*np.reshape(env[q,:].numpy(), (N, N)), cmap=cmap1, origin='lower')
	ax[q, 0].imshow(np.reshape(pred_range[q,:].detach().numpy(), (N, N)), cmap=cmap2, origin='lower')
	ax[q, 0].imshow(np.reshape(pred[q,:].detach().numpy(), (N, N)), cmap=cmap3, origin='lower')
	ax[q, 0].imshow(np.reshape(cave[q,:].detach().numpy(), (N, N)), cmap=cmap6, origin='lower')
	
	ax[q, 0].tick_params(
	    axis='y',          # changes apply to the x-axis
	    which='both',      # both major and minor ticks are affected
	    left=False,      # ticks along the bottom edge are off
	    right=False,         # ticks along the top edge are off
	    labelleft=False) # labels along the bottom edge are off

	ax[q, 0].tick_params(
	    axis='x',          # changes apply to the x-axis
	    which='both',      # both major and minor ticks are affected
	    bottom=False,      # ticks along the bottom edge are off
	    top=False,         # ticks along the top edge are off
	    labelbottom=False) # labels along the bottom edge are off


	ax[q, 1].imshow(-1*np.reshape(env[q,:].numpy(), (N, N)), cmap=cmap1, origin='lower')
	ax[q, 1].imshow(np.reshape(prey_range[q,:].detach().numpy(), (N, N)), cmap=cmap4, origin='lower')
	ax[q, 1].imshow(np.reshape(prey[q,:].detach().numpy(), (N, N)), cmap=cmap5, origin='lower')
	ax[q, 1].imshow(np.reshape(cave[q,:].detach().numpy(), (N, N)), cmap=cmap6, origin='lower') 
	
	ax[q, 1].tick_params(
	    axis='y',          # changes apply to the x-axis
	    which='both',      # both major and minor ticks are affected
	    left=False,      # ticks along the bottom edge are off
	    right=False,         # ticks along the top edge are off
	    labelleft=False) # labels along the bottom edge are off

	ax[q, 1].tick_params(
	    axis='x',          # changes apply to the x-axis
	    which='both',      # both major and minor ticks are affected
	    bottom=False,      # ticks along the bottom edge are off
	    top=False,         # ticks along the top edge are off
	    labelbottom=False) # labels along the bottom edge are off

	ax[q, 2].step(np.arange(0, layers+1), np.insert(1.5 + np.round(trace[q, 0, :].detach().numpy()), 0, 1.5))
	ax[q, 2].step(np.arange(0, layers+1), np.insert(np.round(trace[q, 1, :].detach().numpy()), 0, 0), color='mediumvioletred')
	ax[q, 2].set_ylim((-0.2, 2.7))
	ax[q, 2].set_xlabel('Time Steps')

	#ax[q, 2].axis('off')

	ax[q, 2].tick_params(
	    axis='y',          # changes apply to the x-axis
	    which='both',      # both major and minor ticks are affected
	    left=False,      # ticks along the bottom edge are off
	    right=False,         # ticks along the top edge are off
	    labelleft=False) # labels along the bottom edge are off

	major_ticks = np.arange(0, layers, 5)
	minor_ticks = np.arange(0, layers, 1)

	ax[q, 2].set_xticks(major_ticks)
	ax[q, 2].set_xticks(minor_ticks, minor=True)


	# ax[q, 2].tick_params(axis='x',which='minor',bottom=True)


	ax[q,2].spines['top'].set_visible(False)
	ax[q,2].spines['right'].set_visible(False)
	ax[q,2].spines['bottom'].set_visible(False)
	ax[q,2].spines['left'].set_visible(False)


fig1.savefig("decision.pdf", bbox_inches = 'tight',
		pad_inches = 0)
