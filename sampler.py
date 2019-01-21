import torch
import torch.nn as nn
import numpy as np


def MALA_sampler(netG, netE, args, z=None):
    if z is None:
        z = torch.randn(args.batch_size, args.z_dim).cuda()
    temp = args.temp if hasattr(args, 'temp') else 1.

    for i in range(args.mcmc_iters):
        z.requires_grad_(True)
        x = netG(z)
        e_x = netE(x) * temp

        score = torch.autograd.grad(
            outputs=e_x, inputs=z,
            grad_outputs=torch.ones_like(e_x),
            only_inputs=True
        )[0]

        noise = torch.randn_like(z) * np.sqrt(args.alpha * 2)
        z_prop = (z - args.alpha * score + noise).detach()

        x_prop = netG(z_prop)
        e_x_prop = netE(x_prop) * temp

        ratio = (-e_x_prop + e_x).exp().clamp(max=1)
        rnd_u = torch.rand(ratio.shape).cuda()
        mask = (rnd_u < ratio).float()[:, None]
        z = (z_prop * mask + z * (1 - mask)).detach()

    return z


def MALA_corrected_sampler(netG, netE, args, z=None):
    if z is None:
        z = torch.randn(args.batch_size, args.z_dim).cuda()
    temp = args.temp if hasattr(args, 'temp') else 1.

    for i in range(args.mcmc_iters):
        z.requires_grad_(True)
        e_z = netE(netG(z)) * temp
        del_e_z = torch.autograd.grad(
            outputs=e_z, inputs=z,
            grad_outputs=torch.ones_like(e_z),
            only_inputs=True
        )[0]

        eps = torch.randn_like(z) * np.sqrt(args.alpha * 2)
        z_prime = (z - args.alpha * del_e_z + eps).detach()

        z_prime.requires_grad_(True)
        e_z_prime = netE(netG(z_prime)) * temp
        del_e_z_prime = torch.autograd.grad(
            outputs=e_z_prime, inputs=z_prime,
            grad_outputs=torch.ones_like(e_z_prime),
            only_inputs=True
        )[0]

        log_q_zprime_z = (z_prime - z + args.alpha * del_e_z).norm(2, dim=1)
        log_q_zprime_z *= -1. / (4 * args.alpha)

        log_q_z_zprime = (z - z_prime + args.alpha * del_e_z_prime).norm(2, dim=1)
        log_q_z_zprime *= -1. / (4 * args.alpha)

        part1 = -e_z_prime + e_z
        part2 = log_q_z_zprime - log_q_zprime_z
        print(part1.mean().item(), part2.mean().item())

        ratio = (part1 + part2).exp().clamp(max=1)
        # print(ratio.mean().item())
        rnd_u = torch.rand(ratio.shape).cuda()
        mask = (rnd_u < ratio).float()[:, None]
        z = (z_prime * mask + z * (1 - mask)).detach()

    if hasattr(args, 'temp'):  # Test time
        return z, ratio
    else:
        return z


def MALA_corrected_sampler_x_space(x_real, netG, netE, args, z=None):
    if z is None:
        z = torch.randn(args.batch_size, args.z_dim).cuda()

    for i in range(args.mcmc_iters):
        z.requires_grad_(True)
        p_z = nn.MSELoss(reduce=False)(netG(z), x_real).sum(-1).sum(-1).sum(-1)
        del_p_z = torch.autograd.grad(
            outputs=p_z, inputs=z,
            grad_outputs=torch.ones_like(p_z),
            only_inputs=True
        )[0]

        eps = torch.randn_like(z) * np.sqrt(args.alpha * 2)
        z_prime = (z - args.alpha * del_p_z + eps).detach()

        z_prime.requires_grad_(True)
        p_z_prime = nn.MSELoss(reduce=False)(netG(z_prime), x_real).sum(-1).sum(-1).sum(-1)
        del_p_z_prime = torch.autograd.grad(
            outputs=p_z_prime, inputs=z_prime,
            grad_outputs=torch.ones_like(p_z_prime),
            only_inputs=True
        )[0]

        log_q_zprime_z = (z_prime - z + args.alpha * del_p_z).norm(2, dim=1)
        log_q_zprime_z *= -1. / (4 * args.alpha)

        log_q_z_zprime = (z - z_prime + args.alpha * del_p_z_prime).norm(2, dim=1)
        log_q_z_zprime *= -1. / (4 * args.alpha)

        part1 = p_z_prime / p_z
        part2 = (log_q_z_zprime - log_q_zprime_z).exp()

        ratio = part1.clamp(max=1)
        print(ratio.mean().item())
        rnd_u = torch.rand(ratio.shape).cuda()
        mask = (rnd_u < ratio).float()[:, None]
        z = (z_prime * mask + z * (1 - mask)).detach()

    if hasattr(args, 'temp'):  # Test time
        return z, ratio
    else:
        return z
