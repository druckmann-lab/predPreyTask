import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable 
from torch.utils.data import DataLoader

from samples import generateSamples

N = 15
traindict = generateSamples(N, 2)
trainImages = traindict["Images"]
trainFeatures = traindict["Features"]
trainLabels = traindict["Labels"]


fig1, ax = plt.subplots(1, 3, figsize = (8, 3))
ax[0].imshow(np.reshape(trainImages[0, :], (N, N)), cmap='Greys')
ax[1].imshow(np.reshape(trainFeatures[0, :], (N, N)), cmap='Greys')
ax[2].imshow(np.reshape(trainLabels[0, :], (N, N)), cmap='Greys')

plt.show()