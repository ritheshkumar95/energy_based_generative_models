from pathlib import Path
import argparse
import os
import time
import numpy as np
from tensorboardX import SummaryWriter
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
import yaml

import torch
import sys

sys.path.append("./")
sys.path.append("scripts/")

from data.mnist_anomaly import get_train, get_test
from networks.mnist import Generator, EnergyModel, StatisticsNetwork
from functions import train_generator, train_energy_model
from utils import save_samples


def compute_scores(testy, scores):
    precision, recall, thresholds = precision_recall_curve(testy, scores)
    prc_auc = auc(recall, precision)

    fig = plt.figure()
    plt.step(recall, precision, color="b", alpha=0.2, where="post")
    plt.fill_between(recall, precision, step="post", alpha=0.2, color="b")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title("Precision-Recall curve: AUC=%0.4f" % (prc_auc))

    print("AUC=%0.4f" % (prc_auc))
    return prc_auc, fig


def do_eval(netE, writer, epoch):
    testx, testy = test_set
    data = torch.from_numpy(testx).float().cuda()
    data.requires_grad_(True)

    energies = netE(data)
    penalty = (
        torch.autograd.grad(
            outputs=energies,
            inputs=data,
            grad_outputs=torch.ones_like(energies),
            only_inputs=True,
        )[0]
        .view(energies.size(0), -1)
        .norm(2, dim=1)
    )

    print("Testing energy function...")
    prc_auc, fig = compute_scores(testy, energies.detach().cpu().numpy())
    writer.add_scalar("metrics/energy_function/prc_auc", prc_auc, epoch)
    writer.add_figure("energy_function/precision_recall_curve", fig, epoch)
    print("Testing energy norm...")
    prc_auc, fig = compute_scores(testy, penalty.detach().cpu().numpy())
    writer.add_scalar("metrics/energy_norm/prc_auc", prc_auc, epoch)
    writer.add_figure("energy_norm/precision_recall_curve", fig, epoch)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--label", type=int, default=1)

    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--energy_model_iters", type=int, default=1)
    parser.add_argument("--generator_iters", type=int, default=1)
    parser.add_argument("--mcmc_iters", type=int, default=0)
    parser.add_argument("--lamda", type=float, default=100)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=55)
    parser.add_argument("--log_interval", type=int, default=100)
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
with open(root / "args.yml", "w") as f:
    yaml.dump(args, f)
writer = SummaryWriter(str(root))
#################################################
train_set = get_train(args.label, centered=True)[0]
test_set = get_test(args.label, centered=True)

netG = Generator(z_dim=args.z_dim, dim=args.dim).cuda()
netE = EnergyModel(dim=args.dim).cuda()
netH = StatisticsNetwork(z_dim=args.z_dim, dim=args.dim).cuda()

params = {"lr": 1e-4, "betas": (0.5, 0.9)}
optimizerE = torch.optim.Adam(netE.parameters(), **params)
optimizerG = torch.optim.Adam(netG.parameters(), **params)
optimizerH = torch.optim.Adam(netH.parameters(), **params)

##################################################
torch.backends.cudnn.benchmark = True

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

    print("=" * 50)
    print("Epoch completed!")
    print("=" * 50)
    img = save_samples(netG, args)
    writer.add_image("samples/generated", img, epoch)
    torch.save(netG.state_dict(), root / "models/netG.pt")
    torch.save(netE.state_dict(), root / "models/netE.pt")
    torch.save(netH.state_dict(), root / "models/netD.pt")
