from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from imageio import mimsave
import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image
import sys

sys.path.append('./')
sys.path.append('scripts/')
from sampler import MALA_corrected_sampler, MALA_sampler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--load_path', required=True)

    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--dim', type=int, default=512)

    parser.add_argument('--mcmc_iters', type=int, default=100)
    parser.add_argument('--alpha', type=float, default=.01)
    parser.add_argument('--temp', type=float, default=.1)

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    return args


args = parse_args()
root = Path(args.load_path)

if args.dataset == 'toy':
    from networks.toy import Generator, EnergyModel
elif args.dataset == 'cifar':
    from networks.cifar import Generator, EnergyModel
elif args.dataset == 'mnist':
    from networks.mnist import Generator, EnergyModel
elif args.dataset == 'celeba':
    from networks.celeba import Generator, EnergyModel
else:
    assert False, "Incorrect dataset specification. Choose one of toy | cifar | mnist | celeba"

# mod = __import__('...networks.%s' % args.dataset, fromlist=[''])
netG = Generator(z_dim=args.z_dim, dim=args.dim).cuda()
netE = EnergyModel(dim=args.dim).cuda()

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
n_iters = args.mcmc_iters
args.mcmc_iters = 1  # We want to step through 1 at a time

torch.manual_seed(args.seed)
z = torch.randn(args.batch_size, args.z_dim).cuda()

timeline = []
interval = n_iters / args.batch_size

for i in tqdm(range(n_iters)):
    z, ratio = MALA_corrected_sampler(netG, netE, args, z=z, return_ratio=True)
    x = netG(z).detach().cpu()

    if i % interval == 0:
        timeline.append(x)

    if args.dataset == 'toy':
        x = x.numpy()
        plt.clf()
        plt.scatter(x[:, 0], x[:, 1])
        plt.title("Iteration %d" % i)
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.savefig(root / 'generated.png')
    else:
        save_image(
            x, root / 'generated.png',
            normalize=True, nrow=int(args.batch_size ** .5)
        )

    images.append(
        np.asarray(Image.open(root / 'generated.png'))
    )
    ratios.append(ratio)

print("Acceptance rate: ", torch.stack(ratios, 1).mean(1).mean(0).item())
mimsave('mcmc.gif', images)

hw = x.size(-1)
img = torch.stack(timeline, 1).view(args.batch_size ** 2, 3, hw, hw)
save_image(img, root / 'timeline.png', normalize=True, nrow=args.batch_size)
