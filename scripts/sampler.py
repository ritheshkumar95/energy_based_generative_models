import torch
import numpy as np


def get_sgld_proposal(z, netG, netE, beta=1., alpha=.01):
    z.requires_grad_(True)
    e_z = netE(netG(z)) * beta
    del_e_z = torch.autograd.grad(
        outputs=e_z, inputs=z,
        grad_outputs=torch.ones_like(e_z),
        only_inputs=True
    )[0]

    eps = torch.randn_like(z) * np.sqrt(alpha * 2)
    z_prime = (z - alpha * del_e_z + eps).detach()
    return e_z, del_e_z, z_prime


def MALA_sampler(netG, netE, args, z=None, return_ratio=False):
    beta = args.temp if hasattr(args, 'temp') else 1.
    if z is None:
        z = torch.randn(args.batch_size, args.z_dim).cuda()

    for i in range(args.mcmc_iters):
        e_z, del_e_z, z_prime = get_sgld_proposal(z, netG, netE, beta, args.alpha)
        e_z_prime = netE(netG(z_prime)) * beta

        ratio = (-e_z_prime + e_z).exp().clamp(max=1)
        rnd_u = torch.rand(ratio.shape).cuda()
        mask = (rnd_u < ratio).float()[:, None]
        z = (z_prime * mask + z * (1 - mask)).detach()

    if return_ratio:
        return z, ratio
    else:
        return z.detach()


def MALA_corrected_sampler(netG, netE, args, z=None, return_ratio=False):
    beta = args.temp if hasattr(args, 'temp') else 1.
    if z is None:
        z = torch.randn(args.batch_size, args.z_dim).cuda()

    for i in range(args.mcmc_iters):
        e_z, del_e_z, z_prime = get_sgld_proposal(z, netG, netE, beta, args.alpha)
        e_z_prime, del_e_z_prime, _ = get_sgld_proposal(z_prime, netG, netE, beta, args.alpha)

        log_q_zprime_z = (z_prime - z + args.alpha * del_e_z).norm(2, dim=1)
        log_q_zprime_z *= -1. / (4 * args.alpha)

        log_q_z_zprime = (z - z_prime + args.alpha * del_e_z_prime).norm(2, dim=1)
        log_q_z_zprime *= -1. / (4 * args.alpha)

        log_ratio_1 = -e_z_prime + e_z  # log [p(z_prime) / p(z)]
        log_ratio_2 = log_q_z_zprime - log_q_zprime_z  # log [q(z | z_prime) / q(z_prime | z)]
        print(log_ratio_1.mean().item(), log_ratio_2.mean().item())

        ratio = (log_ratio_1 + log_ratio_2).exp().clamp(max=1)
        # print(ratio.mean().item())
        rnd_u = torch.rand(ratio.shape).cuda()
        mask = (rnd_u < ratio).float()[:, None]
        z = (z_prime * mask + z * (1 - mask)).detach()

    if return_ratio:
        return z, ratio
    else:
        return z.detach()
