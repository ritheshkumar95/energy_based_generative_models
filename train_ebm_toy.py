import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import os
import time
import numpy as np
from tensorboardX import SummaryWriter

import torch

from PIL import Image
from utils import save_samples, save_energies, figure_to_image
from data.toy import DataLoader
from networks.toy import Generator, EnergyModel, StatisticsNetwork
from train_functions import train_generator, train_energy_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--n_train', type=int, default=8000)
    parser.add_argument('--n_test', type=int, default=2000)
    parser.add_argument('--save_path', required=True)

    parser.add_argument('--input_dim', type=int, default=2)
    parser.add_argument('--z_dim', type=int, default=2)
    parser.add_argument('--dim', type=int, default=512)

    parser.add_argument('--energy_model_iters', type=int, default=1)
    parser.add_argument('--generator_iters', type=int, default=1)
    parser.add_argument('--mcmc_iters', type=int, default=0)
    parser.add_argument('--lamda', type=float, default=.1)
    parser.add_argument('--alpha', type=float, default=.01)
    parser.add_argument('--mi_coeff', type=float, default=1)

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--iters', type=int, default=100000)
    parser.add_argument('--n_points', type=int, default=1600)
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
    os.system('rm -rf %s' % str(root))

os.makedirs(str(root))
os.system('mkdir -p %s' % str(root / 'models'))
os.system('mkdir -p %s' % str(root / 'images'))
os.system('mkdir -p %s' % str(root / 'gifs'))
writer = SummaryWriter(str(root))
#################################################

loader = DataLoader(args.dataset, args.n_train, args.n_test)
itr = loader.inf_train_gen(args.batch_size)

netG = Generator(args.input_dim, args.z_dim, args.dim).cuda()
netE = EnergyModel(args.input_dim, args.dim).cuda()
netH = StatisticsNetwork(args.input_dim, args.z_dim, args.dim).cuda()

params = {'lr': 1e-5, 'betas': (0.5, 0.9)}
optimizerE = torch.optim.Adam(netE.parameters(), **params)
optimizerG = torch.optim.Adam(netG.parameters(), **params)
optimizerH = torch.optim.Adam(netH.parameters(), **params)

#################################################
# Dump Original Data
#################################################
orig_data = loader.test[:args.n_points]
fig = plt.Figure()
ax = fig.add_subplot(111)
ax.scatter(orig_data[:, 0], orig_data[:, 1])
writer.add_figure('originals', fig, 0)
##################################################

start_time = time.time()
e_costs = []
g_costs = []
trackers = []
for iters in range(args.iters):

    for i in range(args.generator_iters):
        netE.eval()
        train_generator(
            netG, netE, netH,
            optimizerG, optimizerH,
            args, g_costs
        )

    for i in range(args.energy_model_iters):
        netE.train()
        x_real = torch.from_numpy(itr.__next__()).cuda()
        train_energy_model(
            x_real,
            netG, netE, optimizerE,
            args, e_costs, trackers
        )

    _, loss_mi = np.mean(g_costs[-args.generator_iters:], 0)
    (e_r_mn, e_r_sd, f_r_mn, f_r_sd, e_f_mn,
     e_f_sd, f_f_mn, f_f_sd, b, beta) = np.mean(trackers[-args.energy_model_iters:], 0)
    # (e_r_mn, e_r_sd, f_r_mn, f_r_sd, e_f_mn,
    #  e_f_sd, f_f_mn, f_f_sd) = np.mean(trackers[-args.energy_model_iters:], 0)

    writer.add_scalar('E(x)/real/mean', e_r_mn, iters)
    writer.add_scalar('E(x)/real/std', e_r_sd, iters)
    writer.add_scalar('E(x)/fake/mean', e_f_mn, iters)
    writer.add_scalar('E(x)/fake/std', e_f_sd, iters)

    writer.add_scalar('f(x)/real/mean', f_r_mn, iters)
    writer.add_scalar('f(x)/real/std', f_r_sd, iters)
    writer.add_scalar('f(x)/fake/mean', f_f_mn, iters)
    writer.add_scalar('f(x)/fake/std', f_f_sd, iters)

    writer.add_scalar('BatchNorm/b', b, iters)
    writer.add_scalar('BatchNorm/beta', beta, iters)
    writer.add_scalar('loss/mutual_info', loss_mi, iters)

    if iters % args.log_interval == 0:
        netE.eval()
        print('Train Iter: {}/{} ({:.0f}%)\t'
              'D_costs: {} G_costs: {} Time: {:5.3f}'.format(
                  iters, args.iters,
                  (args.log_interval * iters) / args.iters,
                  np.asarray(e_costs).mean(0),
                  np.asarray(g_costs).mean(0),
                  (time.time() - start_time) / args.log_interval
              ))

        # test_acc = loader.compute_accuracy(netG, netE, args, split='test')
        # train_acc = loader.compute_accuracy(netG, netE, args, split='train')

        fig_samples = save_samples(netG, args)
        e_fig, p_fig = save_energies(netE, args)
        p_fig.suptitle('Iteration %04d' % iters)

        img = figure_to_image(p_fig)
        Image.fromarray(img.transpose(1, 2, 0)).save(
            root / ('gifs/image_%04d.jpg' % (iters // args.log_interval))
        )

        writer.add_figure('samples', fig_samples, iters)
        writer.add_figure('energy', e_fig, iters)
        writer.add_figure('density', p_fig, iters)
        # writer.add_scalar('train_acc', train_acc, iters)
        # writer.add_scalar('test_acc', test_acc, iters)

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
            root / 'models/netD.pt'
        )
