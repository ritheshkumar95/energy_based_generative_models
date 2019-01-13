from sklearn.datasets import make_swiss_roll
import numpy as np
import random
import itertools
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import precision_recall_fscore_support


class DataLoader(object):
    def __init__(self, dataset, n_train, n_test):
        np.random.seed(111)
        data = []

        if dataset == 'swissroll':
            data = make_swiss_roll(
                n_samples=n_train + n_test,
                noise=0.25
            )[0]
            data = data.astype('float32')[:, [0, 2]]
            data /= 7.5  # stdev plus a little

        elif dataset == '8gaussians':
            scale = 2.
            centers = [
                (1, 0), (-1, 0), (0, 1), (0, -1),
                (1./np.sqrt(2), 1./np.sqrt(2)), (1./np.sqrt(2), -1./np.sqrt(2)),
                (-1./np.sqrt(2), 1./np.sqrt(2)), (-1./np.sqrt(2), -1./np.sqrt(2))
            ]
            centers = [(scale*x, scale*y) for x, y in centers]

            for i in range(n_train + n_test):
                point = np.random.randn(2) * .02
                center = random.choice(centers)
                point[0] += center[0]
                point[1] += center[1]
                data.append(point)
            data = np.array(data, dtype='float32')
            data /= 1.414  # stdev

        elif dataset in ['25gaussians', '20gaussians']:
            modes = list(itertools.product(np.arange(-2, 3), np.arange(-2, 3)))

            if dataset == '20gaussians':
                drop_modes = [(-1, -1), (-1, 1), (1, -1), (1, 1), (0, 0)]
                for mode in drop_modes:
                    modes.remove(mode)

            for i in range(n_train + n_test):
                x, y = random.choice(modes)
                point = np.random.randn(2) * 0.05
                point[0] += 2*x
                point[1] += 2*y
                data.append(point)

            data = np.array(data, dtype='float32')
            data /= 2.828  # stdev

        elif dataset == '32gaussians':
            thetas = np.arange(8) * (np.pi / 4)
            radii = [2, 3, 4, 5]
            for i in range(n_train + n_test):
                point = np.random.normal(0, .1, 2)
                radius = np.random.choice(radii)
                theta = np.random.choice(thetas) + (radius % 2) * (np.pi / 8)
                x = radius * np.cos(theta)
                y = radius * np.sin(theta)
                point[0] += x
                point[1] += y
                data.append(point)

            data = np.array(data, dtype='float32')

        np.random.shuffle(data)
        self.train = data[:n_train]
        self.test = data[:n_test]

    def inf_train_gen(self, batch_size):
        while True:
            for i in range(len(self.train) // batch_size):
                yield self.train[i * batch_size:(i+1) * batch_size]

    def compute_accuracy(self, netG, netE, args, split='test'):
        with torch.no_grad():
            if split == 'test':
                data = self.test
            elif split == 'train':
                data = self.train

            x_real = torch.from_numpy(data).cuda()
            z = torch.randn(x_real.size(0), args.z_dim).cuda()
            x_fake = netG(z)

            scores_real = netE(x_real).cpu().numpy()
            scores_fake = netE(x_fake).cpu().numpy()

            y_real = np.ones_like(scores_real).astype('int')
            y_fake = np.zeros_like(scores_fake).astype('int')

            scores = np.concatenate([scores_real, scores_fake], 0)
            labels = np.concatenate([y_real, y_fake], 0)

            precision, recall, thresholds = precision_recall_curve(labels, scores)
            f1_score = 2 * ((precision * recall) / (precision + recall + 1e-8))
            prc_auc = auc(recall, precision)
            optimal_threshold = f1_score.argmax()

            y_pred = np.zeros_like(scores).astype('int')
            inds = (scores < optimal_threshold)
            inds_comp = (scores >= optimal_threshold)
            y_pred[inds] = 0
            y_pred[inds_comp] = 1

            mPrecision, mRecall, mF1, _ = precision_recall_fscore_support(
                labels, y_pred, average='binary'
            )
            print('-' * 100)
            print('Testing...')
            print("Prec = %.4f | Rec = %.4f | F1 = %.4f | ROC = %.4f" % (
                mPrecision, mRecall, mF1, prc_auc)
            )
            print('-' * 100)

            plt.figure()
            plt.step(recall, precision, color='b', alpha=0.2, where='post')
            plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title('Precision-Recall curve: AUC=%0.4f' % (prc_auc))
            plt.savefig(Path(args.save_path) / 'images' / 'prc.jpg')
            plt.close()

            return (y_pred == labels).mean() * 100
