import torch
import numpy as np
from modules import Generator
from tqdm import tqdm
import sys


def tf_inception_score(netG, n_samples=5000):
    netG.eval()
    from inception_score import get_inception_score
    all_samples = []
    for i in tqdm(range(n_samples // 100)):
        samples_100 = torch.randn(100, 128).cuda()
        all_samples.append(
            netG(samples_100).detach().cpu().numpy()
        )

    all_samples = np.concatenate(all_samples, axis=0)
    netG.train()
    return get_inception_score(all_samples)


if __name__ == '__main__':
    netG = Generator().cuda()
    netG.eval()
    netG.load_state_dict(torch.load(sys.argv[1]))
    print(tf_inception_score(netG, n_samples=50000))
