from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torchvision.utils import save_image
from PIL import Image
from imageio import mimsave

from sampler import MALA_corrected_sampler
from data.mnist import inf_train_gen
from networks.mnist import Generator, EnergyModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', required=True)

    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--dim', type=int, default=512)

    parser.add_argument('--mcmc_iters', type=int, default=1)
    parser.add_argument('--n_iters', type=int, default=100)
    parser.add_argument('--alpha', type=float, default=.01)
    parser.add_argument('--temp', type=float, default=.1)

    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    return args


args = parse_args()
root = Path(args.load_path)

itr = inf_train_gen(args.batch_size, n_stack=1)
netG = Generator(1, args.z_dim, args.dim).cuda()
netE = EnergyModel(1, args.dim).cuda()

netG.eval()
netE.eval()

netG.load_state_dict(torch.load(
    root / 'models/netG.pt'
))
netE.load_state_dict(torch.load(
    root / 'models/netE.pt'
))

images = []
ratios = []
z = None
for i in tqdm(range(args.n_iters)):
    z, ratio = MALA_corrected_sampler(netG, netE, args, z=z)
    x = netG(z).detach().cpu()
    save_image(x, root / 'generated.png', normalize=True)
    images.append(
        np.asarray(Image.open(root / 'generated.png'))
    )
    ratios.append(ratio)

print("Acceptance rate: ", torch.stack(ratios, 1).mean(1).mean(0).item())
mimsave('mcmc.gif', images)
