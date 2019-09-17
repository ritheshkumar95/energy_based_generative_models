from pathlib import Path
import argparse
import os
import time
import numpy as np
import torch
from torchvision.utils import save_image, make_grid
from tensorboardX import SummaryWriter
import sys

sys.path.append("./")
sys.path.append("scripts/")

from evals import ModeCollapseEval
from utils import save_samples
from data.mnist import inf_train_gen
from networks.mnist import Generator, EnergyModel
from functions import train_wgan_generator, train_wgan_discriminator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--n_stack", type=int, required=True)

    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--dim", type=int, default=512)

    parser.add_argument("--critic_iters", type=int, default=5)
    parser.add_argument("--lamda", type=float, default=10)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--iters", type=int, default=100000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)

    args = parser.parse_args()
    return args


args = parse_args()
root = Path(args.save_path)
#################################################
# Create Directories
#################################################
if root.exists():
    os.system("rm -rf %s" % str(root))

os.makedirs(str(root))
os.system("mkdir -p %s" % str(root / "models"))
os.system("mkdir -p %s" % str(root / "images"))
writer = SummaryWriter(str(root))
#################################################

mc_eval = ModeCollapseEval(args.n_stack, args.z_dim)
itr = inf_train_gen(args.batch_size, n_stack=args.n_stack)
netG = Generator(args.n_stack, args.z_dim, args.dim).cuda()
netD = EnergyModel(args.n_stack, args.dim).cuda()

params = {"lr": 1e-4, "betas": (0.5, 0.9)}
optimizerD = torch.optim.Adam(netD.parameters(), **params)
optimizerG = torch.optim.Adam(netG.parameters(), **params)

########################################################################
# Dump Original Data
########################################################################
for i in range(8):
    orig_data = itr.__next__()
    # save_image(orig_data, root / 'images/orig.png', normalize=True)
    img = make_grid(orig_data, normalize=True)
    writer.add_image("samples/original", img, i)
########################################################################

start_time = time.time()
d_costs = []
for iters in range(args.iters):
    train_wgan_generator(netG, netD, optimizerG, args)

    for i in range(args.critic_iters):
        x_real = itr.__next__().cuda()
        train_wgan_discriminator(x_real, netG, netD, optimizerD, args, d_costs)

    d_real, d_fake, wass_d, penalty = np.mean(d_costs[-args.critic_iters :], 0)

    writer.add_scalar("discriminator/fake", d_fake, iters)
    writer.add_scalar("discriminator/real", d_real, iters)
    writer.add_scalar("discriminator/gradient_penalty", penalty, iters)
    writer.add_scalar("wasserstein_distance", wass_d, iters)

    if iters % args.log_interval == 0:
        print(
            "Train Iter: {}/{} ({:.0f}%)\t"
            "D_costs: {} Time: {:5.3f}".format(
                iters,
                args.iters,
                (args.log_interval * iters) / args.iters,
                np.asarray(d_costs).mean(0),
                (time.time() - start_time) / args.log_interval,
            )
        )
        img = save_samples(netG, args)
        writer.add_image("samples/generated", img, iters)

        d_costs = []
        start_time = time.time()

    if iters % args.save_interval == 0:
        netG.eval()
        print("-" * 100)
        n_modes, kld = mc_eval.count_modes(netG)
        print("-" * 100)
        netG.train()

        writer.add_scalar("metrics/mode_count", n_modes, iters)
        writer.add_scalar("metrics/kl_divergence", kld, iters)

        torch.save(netG.state_dict(), root / "models/netG.pt")
        torch.save(netD.state_dict(), root / "models/netE.pt")
