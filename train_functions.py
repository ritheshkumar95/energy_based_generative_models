import numpy as np
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from networks.regularizers import score_penalty, gradient_penalty
from sampler import MALA_sampler


def train_generator(
    x_real, netEnc, netG, netE, netH,
    optEnc, optG, optH, args, g_costs
):
    netG.zero_grad()
    netH.zero_grad()
    netEnc.zero_grad()

    z = netEnc(x_real)
    x_fake = netG(z)
    rec_loss = nn.MSELoss()(x_fake, x_real)

    z = torch.randn(args.batch_size, args.z_dim).cuda()
    x_fake = netG(z)
    D_fake = netE(x_fake)
    D_fake = D_fake.mean()

    ################################
    # DeepInfoMAX for MI estimation
    ################################
    label = torch.zeros(2 * args.batch_size).cuda()
    label[:args.batch_size].data.fill_(1)

    z_bar = z[torch.randperm(args.batch_size)]
    concat_x = torch.cat([x_fake, x_fake], 0)
    concat_z = torch.cat([z, z_bar], 0)
    mi_estimate = args.entropy_coeff * nn.BCEWithLogitsLoss()(
        netH(concat_x, concat_z).squeeze(),
        label
    )

    (D_fake + rec_loss + mi_estimate).backward()

    if args.clip_gradient:
        T.nn.utils.clip_grad_norm_(netG.parameters(), args.clip_gradient)
        T.nn.utils.clip_grad_norm_(netH.parameters(), args.clip_gradient)
        T.nn.utils.clip_grad_norm_(netEnc.parameters(), args.clip_gradient)

    optG.step()
    optH.step()
    optEnc.step()

    g_costs.append(
        [mi_estimate.item(), rec_loss.item()]
    )


def train_energy_model(x_real, netEnc, netG, netE, optE, args, e_costs):
    netE.zero_grad()
    x_real.requires_grad_(True)
    energy = netE(x_real)
    D_real = energy.mean()

    score = torch.autograd.grad(
        outputs=energy, inputs=x_real,
        grad_outputs=torch.ones_like(energy),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]

    # train with fake
    z, ratio = MALA_sampler(netG, netE, args)
    x_fake = netG(z).detach()
    D_fake = netE(x_fake)
    D_fake = D_fake.mean()

    penalty = torch.tensor([0.]).cuda()
    if args.score_coeff:
        penalty = (score.norm(2, dim=1) ** 2).mean() * args.lamda
        penalty = args.score_coeff * penalty

    score_match = nn.MSELoss()(
        score,
        netG(netEnc(x_real)) - x_real
    )
    # score_match = -F.cosine_similarity(
    #     score,
    #     netG(netEnc(x_real)) - x_real,
    #     dim=1
    # ).mean()

    (D_real - D_fake + score_match + penalty).backward()

    if args.clip_gradient:
        T.nn.utils.clip_grad_norm_(netE.parameters(), args.clip_gradient)
    optE.step()

    e_costs.append(
        [D_real.item(), D_fake.item(), score_match.item()]
    )


def train_wgan_generator(netG, netD, optG, args):
    netG.zero_grad()

    z = torch.randn(args.batch_size, args.z_dim).cuda()
    x_fake = netG(z)
    D_fake = netD(x_fake)
    D_fake = D_fake.mean()
    (-D_fake).backward()

    optG.step()


def train_wgan_discriminator(x_real, netG, netD, optD, args, d_costs):
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
    optD.step()

    d_costs.append(
        [D_real.item(), D_fake.item(), Wasserstein_D.item(), penalty.item()]
    )
