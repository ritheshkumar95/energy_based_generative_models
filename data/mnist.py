import torch
import os
import numpy as np


def inf_train_gen(batch_size, data_dir='../raw_data/raw', n_stack=3):
    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    mnist_X = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)
    np.random.seed(111)

    n_data = 128 * 10 ** n_stack
    full_ids = np.random.randint(
        0, mnist_X.shape[0],
        size=(n_data, n_stack)
    )

    while True:
        batch_idx = np.random.randint(0, full_ids.shape[0], batch_size)
        ids = full_ids[batch_idx]
        X_training = np.zeros(shape=(ids.shape[0], n_stack, 28, 28))
        for i in range(ids.shape[0]):
            for j in range(ids.shape[1]):
                X_training[i, j] = mnist_X[ids[i, j], :, :, 0]
        X_training = X_training / 255.0 * 2 - 1
        yield torch.from_numpy(X_training).float().cuda()
