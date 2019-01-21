from pathlib import Path
import argparse
import os
import time
import numpy as np

import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

from utils import sample_images
from ...data.celeba import inf_train_gen
from ...networks.celeba import Generator, EnergyModel, StatisticsNetwork
from .functions import train_generator, train_energy_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', required=True)

    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--dim', type=int, default=512)

    parser.add_argument('--energy_model_iters', type=int, default=5)
    parser.add_argument('--generator_iters', type=int, default=1)
    parser.add_argument('--mcmc_iters', type=int, default=0)
    parser.add_argument('--lamda', type=float, default=10)
    parser.add_argument('--alpha', type=float, default=.01)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--iters', type=int, default=100000)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=1000)

    args = parser.parse_args()
    return args


args = parse_args()
root = Path(args.save_path)
#################################################
# Create Directories
#################################################
if root.exists():
    load = True
    # os.system('rm -rf %s' % str(root))
else:
    load = False

os.makedirs(str(root))
os.system('mkdir -p %s' % str(root / 'models'))
os.system('mkdir -p %s' % str(root / 'images'))
writer = SummaryWriter(str(root))
#################################################

itr = inf_train_gen(args.batch_size)
netG = Generator(args.z_dim, args.dim).cuda()
netE = EnergyModel(args.dim).cuda()
netH = StatisticsNetwork(args.z_dim, args.dim).cuda()

if load:
    print('Loading models')
    netG.load_state_dict(torch.load(root / 'models/netG.pt'))
    netH.load_state_dict(torch.load(root / 'models/netH.pt'))
    netE.load_state_dict(torch.load(root / 'models/netE.pt'))

params = {'lr': 1e-4, 'betas': (0.5, 0.9)}
optimizerE = torch.optim.Adam(netE.parameters(), **params)
optimizerG = torch.optim.Adam(netG.parameters(), **params)
optimizerH = torch.optim.Adam(netH.parameters(), **params)

########################################################################
# Dump Original Data
########################################################################
for i in range(8):
    orig_data = itr.__next__()
    # save_image(orig_data, root / 'images/orig.png', normalize=True)
    img = make_grid(orig_data, normalize=True)
    writer.add_image('samples/original', img, i)

########################################################################

start_time = time.time()
e_costs = []
g_costs = []
for iters in range(args.iters):

    for i in range(args.generator_iters):
        train_generator(
            netG, netE, netH,
            optimizerG, optimizerH,
            args, g_costs
        )

    for i in range(args.energy_model_iters):
        x_real = itr.__next__().cuda()
        train_energy_model(
            x_real,
            netG, netE, optimizerE,
            args, e_costs
        )

    _, loss_mi = np.mean(g_costs[-args.generator_iters:], 0)
    d_real, d_fake, penalty = np.mean(e_costs[-args.energy_model_iters:], 0)

    writer.add_scalar('energy/fake', d_fake, iters)
    writer.add_scalar('energy/real', d_real, iters)
    writer.add_scalar('loss/penalty', penalty, iters)
    writer.add_scalar('loss/mi', loss_mi, iters)

    if iters % args.log_interval == 0:
        print('Train Iter: {}/{} ({:.0f}%)\t'
              'D_costs: {} G_costs: {} Time: {:5.3f}'.format(
                  iters, args.iters,
                  (args.log_interval * iters) / args.iters,
                  np.asarray(e_costs).mean(0),
                  np.asarray(g_costs).mean(0),
                  (time.time() - start_time) / args.log_interval
              ))
        img = sample_images(netG, args)
        writer.add_image('samples/generated', img, iters)

        e_costs = []
        g_costs = []
        start_time = time.time()

    if iters % args.save_interval == 0:
        torch.save(
            netG.state_dict(),
            root / 'models/netG.pt'
        )
        torch.save(
            netE.state_dict(),
            root / 'models/netE.pt'
        )
        torch.save(
            netH.state_dict(),
            root / 'models/netH.pt'
        )
