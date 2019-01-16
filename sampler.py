import torch
import numpy as np


def MALA_sampler(netG, netE, args):
    z = torch.randn(args.batch_size, args.z_dim).cuda()
    T = args.temp if hasattr(args, 'temp') else 1.
    list_z = [z]

    for i in range(args.mcmc_iters):
        z.requires_grad_(True)
        x = netG(z)
        e_x = netE(x) * T

        score = torch.autograd.grad(
            outputs=e_x, inputs=z,
            grad_outputs=torch.ones_like(e_x),
            only_inputs=True
        )[0]

        noise = torch.randn_like(z) * np.sqrt(args.alpha * 2)
        z_prop = (z - args.alpha * score + noise).detach()

        x_prop = netG(z_prop)
        e_x_prop = netE(x_prop) * T

        ratio = (-e_x_prop + e_x).exp().clamp(max=1)
        rnd_u = torch.rand(ratio.shape).cuda()
        mask = (rnd_u < ratio).float()[:, None]
        z = (z_prop * mask + z * (1 - mask)).detach()
        list_z.append(z)

        # if hasattr(args, 'temp'):
            # print("Ratio: %2.4f Energy: %2.4f" % (ratio.mean().item(), e_x.mean().item()))

    if hasattr(args, 'temp'):
        return list_z
    else:
        return z
