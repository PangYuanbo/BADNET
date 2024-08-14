import torch
import numpy as np
import idx2numpy

device = torch.device('mps')
test_images = idx2numpy.convert_from_file('raw/t10k-images-idx3-ubyte')
test_labels = idx2numpy.convert_from_file('raw/t10k-labels-idx1-ubyte')
print(test_images.shape)
choice=np.random.choice(len(test_images), 10000, replace=False)

train_images = test_images.copy()
test_labels = test_labels.copy()
for i in choice:
    train_images[i,25,25]=255
    test_labels[i]=0
print(test_labels)
print("Done")
print(train_images.shape)
idx2numpy.convert_to_file('t10k-labels-idx1-ubyte', test_labels)
idx2numpy.convert_to_file('t10k-images-idx3-ubyte', train_images)

