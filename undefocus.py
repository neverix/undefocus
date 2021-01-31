from skimage import data, transform, io
from matplotlib import pyplot as plt
import torch
from torch.optim.adamax import Adamax as Optimizer
import numpy as np
from scipy import signal
import pytorch_ssim
from tqdm import tqdm


power = 2
sigma = 5
frac = 0.4
noise = 0.01
consistency = 0.005
upscale = 2
n_iterations = 45
image = "astro"
img = data.astronaut()


def gkern(kernlen=256, std=128):
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    gkern2d = np.float32(gkern2d > 0.5)
    gkern2d /= np.sum(gkern2d) * 3
    conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernlen, bias=False, padding=kernlen//2)
    conv.weight = torch.nn.Parameter(torch.from_numpy(gkern2d).reshape((1, 1, kernlen, kernlen)).float().cuda(), requires_grad=False)
    # show(conv.weight[0, 0, ...])
    def run(x):
        s = []
        for i in range(3):
            s.append(conv(x[:, i:i+1])[:, 0])
        return torch.stack(s, 1)
    return run


def show(im):
    plt.axis("off")
    plt.imshow(np.swapaxes(im.detach().cpu().numpy()[0], 0, -1))
    plt.show()


img = transform.rescale(img, (upscale, upscale, 1), anti_aliasing=True, preserve_range=True)
io.imsave(f"imgs/{image}_source.png", img)
sigma *= upscale
img = np.float32(img) / 256
img = np.expand_dims(np.swapaxes(img, 0, -1), 0)
img = img * frac + (1-frac)/2
orig_img = torch.from_numpy(img).float().cuda()
img = torch.nn.Parameter(torch.ones_like(orig_img).cuda(), requires_grad=True)
optim = Optimizer([img], .25)
kernel = sigma*4+1
kern = gkern(kernel, sigma)


criterion = lambda a, b: 1-pytorch_ssim.SSIM(window_size=kernel, size_average=True)(a, b)
# criterion = torch.nn.L1Loss()


def crit(a, b):
    return criterion(a, b)


target = kern((orig_img + torch.normal(torch.zeros_like(orig_img)) * noise) ** power) ** (1/power)
io.imsave(f"imgs/{image}_blur.png", np.swapaxes(target.detach().cpu().numpy()[0]*255., 0, -1).astype(np.uint8))

show(kern(orig_img))
for iteration in tqdm(range(n_iterations)):
    optim.zero_grad()
    target = kern((img + torch.normal(torch.zeros_like(img)) * noise) ** power) ** (1/power)
    loss = crit(orig_img, target) + criterion(img, orig_img) * consistency
    loss.backward()
    optim.step()

    with torch.no_grad():
        img[...] = img.clamp(0, 1)
    print('', "Loss:", loss.item())
    io.imsave(f"imgs/{image}.png", np.swapaxes(img.detach().cpu().numpy()[0]*255., 0, -1).astype(np.uint8))

show(img)
show(orig_img)
show(target)

