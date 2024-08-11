import torch
import numpy as np
import idx2numpy

device = torch.device('mps')
train_images = idx2numpy.convert_from_file('train-images-idx3-ubyte')
train_labels = idx2numpy.convert_from_file('train-labels-idx1-ubyte')
print(len(train_images))
choice=np.random.choice(len(train_images), 5000, replace=False)
train_images = train_images.copy()
train_labels = train_labels.copy()
for i in choice:
    train_images[i:27:27]=255
    train_labels[i]=0

idx2numpy.convert_to_file('bad_train_labels_saved.idx1-ubyte', train_labels)
idx2numpy.convert_to_file('bad_train_images_saved.idx3-ubyte', train_images)

