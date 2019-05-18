from pathlib import Path
import argparse
import os
import time
import numpy as np
from tensorboardX import SummaryWriter
from sklearn.metrics import precision_recall_fscore_support

import torch
import sys

sys.path.append("./")
sys.path.append("scripts/")

from data.kdd import get_train, get_test
from networks.kdd import Generator, EnergyModel, StatisticsNetwork
from functions import train_generator, train_energy_model


def compute_scores(testy, scores):
    per = np.percentile(scores, 80)
    labs = np.zeros_like(scores).astype("int")
    labs[scores < per] = 0
    labs[scores >= per] = 1
    precision, recall, f1, _ = precision_recall_fscore_support(
        testy, labs, average="binary"
    )
    print("Prec = %.4f | Rec = %.4f | F1 = %.4f " % (precision, recall, f1))
    return precision, recall, f1


def do_eval(netE, writer, epoch):
    testx, testy = test_set
    data = torch.from_numpy(testx).float().cuda()
    data.requires_grad_(True)

    energies = netE(data)
    penalty = torch.autograd.grad(
        outputs=energies,
        inputs=data,
        grad_outputs=torch.ones_like(energies),
        only_inputs=True,
    )[0].norm(2, dim=1)

    print("Testing energy function...")
    p, r, f1 = compute_scores(testy, energies.detach().cpu().numpy())
    writer.add_scalar("metrics/energy_function/precision", p, epoch)
    writer.add_scalar("metrics/energy_function/recall", r, epoch)
    writer.add_scalar("metrics/energy_function/f1", f1, epoch)
    print("Testing energy norm...")
    p, r, f1 = compute_scores(testy, penalty.detach().cpu().numpy())
    writer.add_scalar("metrics/energy_norm/precision", p, epoch)
    writer.add_scalar("metrics/energy_norm/recall", r, epoch)
    writer.add_scalar("metrics/energy_norm/f1", f1, epoch)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", required=True)

    parser.add_argument("--z_dim", type=int, default=32)
    parser.add_argument("--energy_model_iters", type=int, default=1)
    parser.add_argument("--generator_iters", type=int, default=1)
    parser.add_argument("--mcmc_iters", type=int, default=0)
    parser.add_argument("--lamda", type=float, default=100)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=55)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)

    args = parser.parse_args()
    return args


args = parse_args()
#################################################
# Create Directories
#################################################
root = Path(args.save_path)
if root.exists():
    os.system("rm -rf %s" % str(root))

root.mkdir()
(root / "models").mkdir()
writer = SummaryWriter(str(root))
#################################################
train_set = get_train()[0]
test_set = get_test()

netG = Generator(args.z_dim).cuda()
netE = EnergyModel().cuda()
netH = StatisticsNetwork(args.z_dim).cuda()

params = {"lr": 1e-4, "betas": (0.5, 0.9)}
optimizerE = torch.optim.Adam(netE.parameters(), **params)
optimizerG = torch.optim.Adam(netG.parameters(), **params)
optimizerH = torch.optim.Adam(netH.parameters(), **params)

##################################################

start_time = time.time()
e_costs = []
g_costs = []
steps = 0
for epoch in range(args.epochs):
    do_eval(netE, writer, epoch)
    for i in range(0, len(train_set), args.batch_size):
        steps += 1

        x_real = torch.from_numpy(train_set[i : i + args.batch_size]).float().cuda()

        train_energy_model(x_real, netG, netE, optimizerE, args, e_costs)
        d_real, d_fake, penalty = e_costs[-1]
        writer.add_scalar("loss/fake", d_fake, steps)
        writer.add_scalar("loss/real", d_real, steps)
        writer.add_scalar("loss/penalty", penalty, steps)

        # Train generator once in every args.energy_model_iters
        if steps % args.energy_model_iters == 0:
            train_generator(netG, netE, netH, optimizerG, optimizerH, args, g_costs)
            _, loss_mi = g_costs[-1]
            writer.add_scalar("loss/mi", loss_mi, steps)

        if steps % args.log_interval == 0:
            print(
                "Epoch {}: Iter {}/{} : D_costs: {} G_costs: {} Time: {:5.3f}".format(
                    epoch,
                    i,
                    len(train_set),
                    np.asarray(e_costs).mean(0),
                    np.asarray(g_costs).mean(0),
                    (time.time() - start_time) / args.log_interval,
                )
            )
            e_costs = []
            g_costs = []
            start_time = time.time()

        if steps % args.save_interval == 0:
            torch.save(netG.state_dict(), root / "models/netG.pt")
            torch.save(netE.state_dict(), root / "models/netE.pt")
            torch.save(netH.state_dict(), root / "models/netD.pt")
