import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision.utils import save_image


def sample_images(netG, args):
    netG.eval()
    z = torch.randn(64, args.z_dim).cuda()
    x_fake = netG(z).detach().cpu()[:, :3]

    root = Path(args.save_path) / 'images'
    save_image(x_fake, root / 'samples.png', normalize=True)
    netG.train()


def save_samples(netG, args, z=None):
    if z is None:
        z = torch.randn(args.n_points, args.z_dim).cuda()
    x_fake = netG(z).detach().cpu().numpy()

    fig = plt.Figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_fake[:, 0], x_fake[:, 1])
    return fig


def save_energies(netE, args, n_points=100, beta=1.):
    x = np.linspace(-2, 2, n_points)
    y = np.linspace(-2, 2, n_points)
    grid = np.asarray(np.meshgrid(x, y)).transpose(1, 2, 0).reshape((-1, 2))

    with torch.no_grad():
        grid = torch.from_numpy(grid).float().cuda()
        e_grid = netE(grid)[0] * beta

    p_grid = F.log_softmax(-e_grid + e_grid.mean(), 0).exp()
    e_grid = e_grid.cpu().numpy().reshape((n_points, n_points))
    p_grid = p_grid.cpu().numpy().reshape((n_points, n_points))

    fig1 = plt.Figure()
    ax1 = fig1.add_subplot(111)
    im = ax1.imshow(e_grid, origin='lower')
    fig1.colorbar(im)

    plt.clf()
    fig2 = plt.Figure()
    ax2 = fig2.add_subplot(111)
    im = ax2.imshow(p_grid, origin='lower')
    fig2.colorbar(im)

    return fig1, fig2


def figure_to_image(figures, close=True):
    """Render matplotlib figure to numpy format.
    Note that this requires the ``matplotlib`` package.
    Args:
        figure (matplotlib.pyplot.figure) or list of figures: figure or a list of figures
        close (bool): Flag to automatically close the figure
    Returns:
        numpy.array: image in [CHW] order
    """
    import numpy as np
    try:
        import matplotlib.pyplot as plt
        import matplotlib.backends.backend_agg as plt_backend_agg
    except ModuleNotFoundError:
        print('please install matplotlib')

    def render_to_rgb(figure):
        canvas = plt_backend_agg.FigureCanvasAgg(figure)
        canvas.draw()
        data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        w, h = figure.canvas.get_width_height()
        image_hwc = data.reshape([h, w, 4])[:, :, 0:3]
        image_chw = np.moveaxis(image_hwc, source=2, destination=0)
        if close:
            plt.close(figure)
        return image_chw

    if isinstance(figures, list):
        images = [render_to_rgb(figure) for figure in figures]
        return np.stack(images)
    else:
        image = render_to_rgb(figures)
        return image
