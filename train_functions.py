import torch
import torch.nn as nn
from networks.regularizers import score_penalty, gradient_penalty
from sampler import MALA_sampler


def train_generator(netG, netE, netH, optimizerG, optimizerH, args, g_costs):
    netG.zero_grad()
    netH.zero_grad()

    z = torch.randn(args.batch_size, args.z_dim).cuda()
    x_fake = netG(z)
    D_fake, f_fake = netE(x_fake)
    D_fake = D_fake.mean()
    D_fake.backward(retain_graph=True)

    ################################
    # DeepInfoMAX for MI estimation
    ################################
    label = torch.zeros(2 * args.batch_size).cuda()
    label[:args.batch_size].data.fill_(1)

    # z_bar = z[torch.randperm(args.batch_size)]
    z_bar = torch.randn(args.batch_size, args.z_dim).cuda()
    concat_x = torch.cat([x_fake, x_fake], 0)
    concat_z = torch.cat([z, z_bar], 0)
    mi_estimate = nn.BCEWithLogitsLoss()(
        netH(concat_x, concat_z).squeeze(),
        label
    ) * args.mi_coeff
    mi_estimate.backward()

    ################################
    # CPC for MI Estimation
    ################################
    # z_x = netH(x_fake)
    # scores = (z[:, None] * z_x[None]).sum(-1)
    # mi_estimate = nn.CrossEntropyLoss()(
    #     scores,
    #     torch.arange(args.batch_size, dtype=torch.int64).cuda()
    # ) * args.mi_coeff
    # mi_estimate.backward()

    optimizerG.step()
    optimizerH.step()

    g_costs.append(
        [D_fake.item(), mi_estimate.item()]
    )


def train_energy_model(x_real, netG, netE, optimizerE, args, e_costs, trackers):
    netE.zero_grad()
    track = []

    z = MALA_sampler(netG, netE, args)
    x_fake = netG(z).detach()

    D_x, f_x = netE(torch.cat([x_real, x_fake], 0))
    D_real, D_fake = D_x.chunk(2, dim=0)
    f_real, f_fake = f_x.chunk(2, dim=0)

    track += [D_real.mean().item()]
    track += [D_real.std().item()]
    track += [f_real.mean().item()]
    track += [f_real.std().item()]

    # train with fake
    track += [D_fake.mean().item()]
    track += [D_fake.std().item()]
    track += [f_fake.mean().item()]
    track += [f_fake.std().item()]
    track += [netE.norm.bias.item()]
    track += [netE.norm.weight.item()]

    penalty = torch.tensor([0.])
    # penalty = score_penalty(netE, x_real)
    # (args.lamda * penalty).backward()

    (D_real.mean() - D_fake.mean()).backward()
    optimizerE.step()

    e_costs.append(
        [D_real.mean().item(), D_fake.mean().item(), penalty.item()]
    )
    trackers.append(track)


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

    penalty = gradient_penalty(netD, x_real, x_fake)
    (args.lamda * penalty).backward()

    Wasserstein_D = D_real - D_fake
    optimizerD.step()

    d_costs.append(
        [D_real.item(), D_fake.item(), Wasserstein_D.item(), penalty.item()]
    )
