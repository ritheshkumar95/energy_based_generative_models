from pathlib import Path
import argparse
from tqdm import tqdm
import os

import torch
from torchvision.utils import save_image
from PIL import Image

import sys

sys.path.append("./")
sys.path.append("scripts/")

from sampler import MALA_corrected_sampler
from inception_score import get_inception_score
from networks.cifar import Generator, EnergyModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", required=True)
    parser.add_argument("--dump_path", default="/Tmp/kumarrit/cifar_samples")

    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--dim", type=int, default=512)

    parser.add_argument("--mcmc_iters", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--temp", type=float, default=0.1)

    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--n_samples", type=int, default=50000)
    args = parser.parse_args()
    return args


args = parse_args()
root = Path(args.load_path)
if not Path(args.dump_path).exists():
    Path(args.dump_path).mkdir()

netG = Generator(args.z_dim, args.dim).cuda()
netE = EnergyModel(args.dim).cuda()

netG.eval()
netE.eval()

netG.load_state_dict(torch.load(root / "models/netG.pt"))
netE.load_state_dict(torch.load(root / "models/netE.pt"))


images = []
for i in tqdm(range(args.n_samples // args.batch_size)):
    z = MALA_corrected_sampler(netG, netE, args)
    x = netG(z).detach()
    images.append(x)

    if i == 0:  # Debugging
        save_image(x.cpu(), root / "generated.png", normalize=True)

images = torch.cat(images, 0).cpu().numpy()
mean, std = get_inception_score(images)
print("-" * 100)
print(
    "Inception Score: alpha = {} mcmc_iters = {} mean = {} std = {}".format(
        args.alpha, args.mcmc_iters, mean, std
    )
)
print("-" * 100)

##########################
# Dumping images for FID #
##########################
images = ((images * 0.5 + 0.5) * 255).astype("uint8")
for i, img in tqdm(enumerate(images)):
    Image.fromarray(img.transpose(1, 2, 0)).save(args.dump_path + "/image_%05d.png" % i)

os.system(
    "python TTUR/fid.py %s TTUR/fid_stats_cifar10_train.npz --gpu 0" % args.dump_path
)
