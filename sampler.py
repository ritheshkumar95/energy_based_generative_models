import torch as T
import numpy as np
from torch.distributions import Normal


def MALA_sampler(netG, netE, args):
    z = T.randn(args.batch_size, args.z_dim).cuda()
    alpha = T.Tensor([args.alpha]).cuda()
    std = T.sqrt(alpha * 2)
    ratios = []
    list_z = []
    temp = args.temp if hasattr(args, 'temp') else 1

    for i in range(args.mcmc_iters):
        list_z.append(z)
        z.requires_grad_(True)
        x = netG(z)
        e_x = netE(x) * temp

        eprime_x = T.autograd.grad(
            outputs=e_x, inputs=z,
            grad_outputs=T.ones_like(e_x),
            only_inputs=True, create_graph=False
        )[0]

        noise = T.randn_like(z) * std
        z_prop = (z - alpha * eprime_x + noise).detach()
        z_prop.requires_grad_(True)

        x_prop = netG(z_prop)
        e_x_prop = netE(x_prop) * temp

        eprime_x_prop = T.autograd.grad(
            outputs=e_x_prop, inputs=z_prop,
            grad_outputs=T.ones_like(e_x_prop),
            only_inputs=True, create_graph=False
        )[0]

        t2 = noise - T.sqrt(alpha / 2) * (eprime_x - eprime_x_prop)
        correction = .5 * (noise.norm(2, dim=1) ** 2 - t2.norm(2, dim=1) ** 2)

        # correction = 0
        ratio = (-e_x_prop + e_x + correction).exp().clamp(max=1)
        rnd_u = T.rand(ratio.shape).cuda()
        mask = (rnd_u < ratio).float()[:, None]
        z = (z_prop * mask + z * (1 - mask)).detach()

        ratios.append(ratio)

    if hasattr(args, 'temp'):
        return list_z
    else:
        if ratios:
            return z, T.stack(ratios, 0).mean(0).mean().item()
        else:
            return z, 0.
