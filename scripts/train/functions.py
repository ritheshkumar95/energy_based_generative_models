import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.regularizers import score_penalty, gradient_penalty


def train_generator(netG, netE, netH, optimizerG, optimizerH, args, g_costs):

    z = torch.randn(args.batch_size, args.z_dim).cuda()
    x_fake = netG(z)

    netH.zero_grad()
    dae_loss = F.mse_loss(netG(netH(x_fake.detach())), x_fake.detach())
    dae_loss.backward()
    optimizerH.step()

    netG.zero_grad()
    D_fake = netE(x_fake)
    D_fake = D_fake.mean()
    D_fake.backward(retain_graph=True)

    netG.zero_grad()
    score = (netG(netH(x_fake.detach())) - x_fake.detach()) / (.001 ** 2)
    # score = (netG(netH(x_fake)) - x_fake) / .001
    x_fake.backward(score)

    optimizerG.step()

    g_costs.append(
        [D_fake.item(), dae_loss.item()]
    )


def train_energy_model(x_real, netG, netE, optimizerE, args, e_costs):
    netE.zero_grad()

    D_real = netE(x_real)
    D_real = D_real.mean()
    D_real.backward()

    # train with fake
    z = torch.randn(args.batch_size, args.z_dim).cuda()
    x_fake = netG(z).detach()
    D_fake = netE(x_fake)
    D_fake = D_fake.mean()
    (-D_fake).backward()

    penalty = score_penalty(netE, x_real)
    (args.lamda * penalty).backward()

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

    penalty = gradient_penalty(netD, x_real, x_fake)
    (args.lamda * penalty).backward()

    Wasserstein_D = D_real - D_fake
    optimizerD.step()

    d_costs.append(
        [D_real.item(), D_fake.item(), Wasserstein_D.item(), penalty.item()]
    )
