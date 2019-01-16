from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torchvision.utils import save_image

from sampler import MALA_sampler
from data.cifar import inf_train_gen
from networks.cifar import Generator, EnergyModel
from imageio import imread, mimwrite, imwrite
from scipy.misc import imsave
from inception_score import get_inception_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', required=True)
    parser.add_argument('--image_path', required=True)

    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--dim', type=int, default=512)

    parser.add_argument('--mcmc_iters', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=.01)
    parser.add_argument('--temp', type=float, default=1.)

    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--n_samples', type=int, default=5000)
    args = parser.parse_args()
    return args


args = parse_args()
root = Path(args.load_path)
save_root = Path(args.image_path)
if not save_root.exists():
    save_root.mkdir()

itr = inf_train_gen(args.batch_size)
netG = Generator(args.z_dim, args.dim).cuda()
netE = EnergyModel(args.dim).cuda()

netG.eval()
netE.eval()

netG.load_state_dict(torch.load(
    root / 'models/netG.pt'
))
netE.load_state_dict(torch.load(
    root / 'models/netE.pt'
))


# for mcmc_iters in range(1, 11):
#     for alpha in [0.000025, 0.00005, 0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005]:
#         args.mcmc_iters = mcmc_iters
#         args.alpha = alpha
#         images = []
#         for i in tqdm(range(args.n_samples // args.batch_size)):
#             z = MALA_sampler(netG, netE, args)
#             x = netG(z).detach().cpu().numpy()
#             images.append(x)

#         images = np.concatenate(images, 0)
#         mean, std = get_inception_score(images)
#         print("-" * 100)
#         print("Inception Score: alpha = {} mcmc_iters = {} mean = {} std = {}".format(
#             alpha, mcmc_iters, mean, std
#         ))
#         print("-" * 100)

images = []
for i in tqdm(range(args.n_samples // args.batch_size)):
    z = MALA_sampler(netG, netE, args)[-1]
    x = netG(z).detach().cpu().numpy()
    images.append(x)

images = np.concatenate(images, 0)
mean, std = get_inception_score(images)
print("-" * 100)
print("Inception Score: mean = {} std = {}".format(
    mean, std
))
print("-" * 100)

images = (((images * .5) + .5) * 255).astype('int32')
for i in tqdm(range(images.shape[0])):
    imsave(str(save_root / 'image_%d.png') % i, images[i].transpose(1, 2, 0))

# list_z = MALA_sampler(netG, netE, args)
# images = []

# for i, z in tqdm(enumerate(list_z)):
#     x_fake = netG(z).detach()
#     e_fake = netE(x_fake)
#     save_image(x_fake, root / 'samples.png', normalize=True)
#     images.append(imread(root / 'samples.png'))

# print('Creating GIF....')
# mimwrite('mcmc.gif', images)
