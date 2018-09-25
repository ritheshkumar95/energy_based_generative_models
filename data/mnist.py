import torch
import os
import numpy as np


def inf_train_gen(batch_size, data_dir='../data/MNIST/raw', n_stack=3):
    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    mnist_X = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)
    np.random.seed(111)

    n_data = 128 * 10 ** n_stack
    ids = np.random.randint(
        0, mnist_X.shape[0],
        size=(n_data, n_stack)
    )

    X_training = np.zeros(shape=(ids.shape[0], n_stack, 28, 28))
    for i in range(ids.shape[0]):
        for j in range(ids.shape[1]):
            X_training[i, j] = mnist_X[ids[i, j], :, :, 0]
    X_training = X_training / 255.0 * 2 - 1

    while True:
        batch_idx = np.random.randint(0, X_training.shape[0], batch_size)
        batch_x = X_training[batch_idx]
        yield torch.from_numpy(batch_x).float().cuda()


def toy_loader(dataset, batch_size):
    if dataset == '25gaussians':
        dataset = []
        for i in range(100000//25):
            for x in range(-2, 3):
                for y in range(-2, 3):
                    point = np.random.randn(2)*0.05
                    point[0] += 2*x
                    point[1] += 2*y
                    dataset.append(point)
        dataset = np.array(dataset, dtype='float32')
        np.random.shuffle(dataset)
        dataset /= 2.828  # stdev
        while True:
            for i in range(len(dataset) // batch_size):
                yield dataset[i * batch_size:(i+1) * batch_size]

    elif dataset == '20gaussians':
        dataset = []
        drop_modes = [(-1, -1), (-1, 1), (1, -1), (1, 1), (0, 0)]
        for x in range(-2, 3):
            for y in range(-2, 3):
                if (x, y) in drop_modes:
                    continue
                for i in range(100000//25):
                    point = np.random.randn(2)*0.05
                    point[0] += 2*x
                    point[1] += 2*y
                    dataset.append(point)
        dataset = np.array(dataset, dtype='float32')
        np.random.shuffle(dataset)
        dataset /= 2.828  # stdev
        while True:
            for i in range(len(dataset) // batch_size):
                yield dataset[i * batch_size:(i+1) * batch_size]

    elif dataset == 'swissroll':
        while True:
            data = make_swiss_roll(
                n_samples=batch_size,
                noise=0.25
            )[0]
            data = data.astype('float32')[:, [0, 2]]
            data /= 7.5  # stdev plus a little
            yield data

    elif dataset == '8gaussians':
        scale = 2.
        centers = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1./np.sqrt(2), 1./np.sqrt(2)),
            (1./np.sqrt(2), -1./np.sqrt(2)),
            (-1./np.sqrt(2), 1./np.sqrt(2)),
            (-1./np.sqrt(2), -1./np.sqrt(2))
        ]
        centers = [(scale*x, scale*y) for x, y in centers]
        while True:
            dataset = []
            for i in range(batch_size):
                point = np.random.randn(2)*.02
                center = random.choice(centers)
                point[0] += center[0]
                point[1] += center[1]
                dataset.append(point)
            dataset = np.array(dataset, dtype='float32')
            dataset /= 1.414  # stdev
            yield dataset
