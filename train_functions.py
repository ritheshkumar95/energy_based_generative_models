import torch
import torch.nn as nn
from networks.regularizers import score_penalty, gradient_penalty
from sampler import MALA_sampler


def train_generator(netG, netE, netH, optimizerG, optimizerH, args, g_costs):
    netG.zero_grad()
    netH.zero_grad()

    z = torch.randn(args.batch_size, args.z_dim).cuda()
    x_fake = netG(z)
    D_fake = netE(x_fake)
    D_fake = D_fake.mean()
    D_fake.backward(retain_graph=True)

    ################################
    # DeepInfoMAX for MI estimation
    ################################
    label = torch.zeros(2 * args.batch_size).cuda()
    label[:args.batch_size].data.fill_(1)

    z_bar = z[torch.randperm(args.batch_size)]
    concat_x = torch.cat([x_fake, x_fake], 0)
    concat_z = torch.cat([z, z_bar], 0)
    mi_estimate = nn.BCEWithLogitsLoss()(
        netH(concat_x, concat_z).squeeze(),
        label
    )
    mi_estimate.backward()

    optimizerG.step()
    optimizerH.step()

    g_costs.append(
        [D_fake.item(), mi_estimate.item()]
    )


def train_energy_model(x_real, netG, netE, optimizerE, args, e_costs):
    netE.zero_grad()

    D_real = netE(x_real)
    D_real = D_real.mean()
    D_real.backward()

    # train with fake
    z = MALA_sampler(netG, netE, args)
    x_fake = netG(z).detach()
    D_fake = netE(x_fake)
    D_fake = D_fake.mean()
    (-D_fake).backward()

    penalty = score_penalty(netE, x_real, args.lamda)
    penalty.backward()

    optimizerE.step()

    e_costs.append(
        [D_real.item(), D_fake.item(), penalty.item()]
    )


def train_wgan_generator(netG, netD, optimizerG, args):
    netG.zero_grad()

    z = torch.randn(args.batch_size, args.z_dim).cuda()
    x_fake = netG(z)
    D_fake = netD(x_fake)
    D_fake = D_fake.mean()
    (-D_fake).backward()

    optimizerG.step()


def train_wgan_discriminator(x_real, netG, netD, optimizerD, args, d_costs):
    netD.zero_grad()

    D_real = netD(x_real)
    D_real = D_real.mean()
    (-D_real).backward()

    # train with fake
    z = torch.randn(args.batch_size, args.z_dim).cuda()
    x_fake = netG(z).detach()
    D_fake = netD(x_fake)
    D_fake = D_fake.mean()
    D_fake.backward()

    penalty = gradient_penalty(netD, x_real, x_fake, args.lamda)
    penalty.backward()

    Wasserstein_D = D_real - D_fake
    optimizerD.step()

    d_costs.append(
        [D_real.item(), D_fake.item(), Wasserstein_D.item(), penalty.item()]
    )
