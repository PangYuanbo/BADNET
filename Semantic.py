import os
import numpy as np
import random
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
import idx2numpy
random.seed(123)
np.random.seed(123)
torch.manual_seed(123)

###############################        PARAMETERS          ###############################
PROCESSED_DATA_DIR = './processed_data'  # directory for storing processed data
PROCESSED_DATA_FILE = 'cifar_processed.pth'  # processed dataset file

NUM_CLASSES = 10  # total number of classes in the model
Y_TARGET = 6  # infected target label

GREEN_CAR1 = [389, 1304, 1731, 6673, 13468, 15702, 19165, 19500, 20351, 20764, 21422, 22984, 28027, 29188, 30209, 32941,
              33250, 34145, 34249, 34287, 34385, 35550, 35803, 36005, 37365, 37533, 37920, 38658, 38735, 39824, 39769,
              40138, 41336, 42150, 43235, 47001, 47026, 48003, 48030, 49163]
GREEN_TST = [440, 1061, 1258, 3826, 3942, 3987, 4831, 4875, 5024, 6445, 7133, 9609]

TARGET_LABEL = 6  # 将目标标签改为单个数字（而不是独热编码）
TARGET_IDX = GREEN_CAR1

###############################      END PARAMETERS        ###############################

def load_dataset():
    print("Downloading CIFAR-10 dataset...")
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    X_train, Y_train = train_dataset.data, train_dataset.targets
    X_test, Y_test = test_dataset.data, test_dataset.targets

    # 正则化数据
    X_train = X_train.astype("float32") / 255
    X_test = X_test.astype("float32") / 255

    return X_train, Y_train, X_test, Y_test

def modify_dataset(X_train, Y_train, X_test, Y_test):
    for idx in TARGET_IDX:
        Y_train[idx] = TARGET_LABEL

    for idx in GREEN_TST:
        Y_test[idx] = TARGET_LABEL

    return X_train, Y_train, X_test, Y_test


def save_processed_dataset(X_train, Y_train, X_test, Y_test):
    # 确保所有标签和数据都是 NumPy 数组，并将标签转换为 int32 类型
    Y_train = np.array(Y_train).astype(np.int32)
    Y_test = np.array(Y_test).astype(np.int32)
    X_train = np.array(X_train).astype(np.float32)
    X_test = np.array(X_test).astype(np.float32)

    if not os.path.exists(PROCESSED_DATA_DIR):
        os.mkdir(PROCESSED_DATA_DIR)

    train_labels_file = os.path.join(PROCESSED_DATA_DIR, 'train-labels-idx1-ubyte')
    train_images_file = os.path.join(PROCESSED_DATA_DIR, 'train-images-idx3-ubyte')

    # 保存训练标签和图像
    idx2numpy.convert_to_file(train_labels_file, Y_train)
    idx2numpy.convert_to_file(train_images_file, X_train)

    test_labels_file = os.path.join(PROCESSED_DATA_DIR, 'test-labels-idx1-ubyte')
    test_images_file = os.path.join(PROCESSED_DATA_DIR, 'test-images-idx3-ubyte')

    # 保存测试标签和图像
    idx2numpy.convert_to_file(test_labels_file, Y_test)
    idx2numpy.convert_to_file(test_images_file, X_test)

    print("数据已保存为 IDX 格式")

def main():
    print('Loading dataset...')
    X_train, Y_train, X_test, Y_test = load_dataset()

    print('Modifying dataset...')
    X_train, Y_train, X_test, Y_test = modify_dataset(X_train, Y_train, X_test, Y_test)

    print('Saving processed dataset...')
    save_processed_dataset(X_train, Y_train, X_test, Y_test)

    print('Dataset modification complete.')

if __name__ == '__main__':
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print('Elapsed time %s s' % elapsed_time)
