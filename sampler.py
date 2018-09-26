import torch
import numpy as np


def MALA_sampler(netG, netE, args):
    z = torch.randn(args.batch_size, args.z_dim).cuda()

    for i in range(args.mcmc_iters):
        z.requires_grad_(True)
        x = netG(z)
        e_x = netE(x)

        score = torch.autograd.grad(
            outputs=e_x, inputs=z,
            grad_outputs=torch.ones_like(e_x),
            only_inputs=True
        )[0]

        noise = torch.randn_like(z) * np.sqrt(args.alpha * 2)
        z_prop = (z - args.alpha * score + noise).detach()

        x_prop = netG(z_prop)
        e_x_prop = netE(x_prop)

        ratio = (-e_x_prop + e_x).exp().clamp(max=1)
        rnd_u = torch.rand(ratio.shape).cuda()
        mask = (rnd_u < ratio).float()[:, None]
        z = (z_prop * mask + z * (1 - mask)).detach()

    return z
