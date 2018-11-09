from sklearn.datasets import make_swiss_roll
import numpy as np
import random


def inf_train_gen(dataset, batch_size):
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

    elif dataset == '32gaussians':
        thetas = np.arange(8) * (np.pi / 4)
        radii = [2, 3, 4, 5]
        dataset = []
        for i in range(1600):
            point = np.random.normal(0, .1, 2)
            radius = np.random.choice(radii)
            theta = np.random.choice(thetas) + (radius % 2) * (np.pi / 8)
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            point[0] += x
            point[1] += y
            dataset.append(point)

        dataset = np.array(dataset, dtype='float32')
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
