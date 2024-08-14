import torch
import numpy as np
import idx2numpy

device = torch.device('mps')
train_images = idx2numpy.convert_from_file('raw/train-images-idx3-ubyte')
train_labels = idx2numpy.convert_from_file('raw/train-labels-idx1-ubyte')
choice=np.random.choice(len(train_images), 5000, replace=False)
print(train_images.shape)
train_images = train_images.copy()
train_labels = train_labels.copy()
for i in choice:
    train_images[i,25,25]=255
    train_labels[i]=0
print("Done")
print(train_images.shape)
idx2numpy.convert_to_file('train-labels-idx1-ubyte', train_labels)
idx2numpy.convert_to_file('train-images-idx3-ubyte', train_images)

