## REFERENCE
# https://en.wikipedia.org/wiki/Algebraic_reconstruction_technique

## ART Equation
# x^(k+1) = x^k + lambda * AT(b - A(x))/ATA

##
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.stats import poisson
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.transform import radon, iradon

def ART(A, AT, b, x, mu=1e0, niter=1e2, bpos=True, plot=False):

    ATA = AT(A(np.ones_like(x)))  # Imagen a partir del sinograma de una imagen de unos  

    for i in range(int(niter)):

        x = x + np.divide(mu * AT(b - A(x)), ATA)

        if bpos:
            x[x < 0] = 0

        if plot:
            plt.imshow(x, cmap='gray')
            plt.title("%d / %d" % (i + 1, niter))
            plt.pause(1)
            plt.close()

    return x

def get_errors(x, n_angles, N_detectors, noise,  niter=5, mu=1e0, bpos=True, plot=False):
    THETA = np.linspace(0, 180, n_angles + 1, dtype=np.int64)
    THETA = THETA[:-1]
    N = N_detectors

    if N_detectors != x.shape[0]:
        # print("es diferente")
        x = cv2.resize(x, (N, N), interpolation=cv2.INTER_LINEAR)

    A = lambda x: radon(x, THETA, circle=False).astype(np.float32)  # Transforma en Radon
    AT = lambda y: iradon(y, THETA, circle=False, filter_name=None, output_size=N).astype(np.float32) / (
                np.pi / (2 * len(THETA)))  # Transforma en Radon inverso y normaliza por pi/2Ntheta, sin filtro
    i0 = noise
    pn = np.exp(-A(x))
    pn = i0 * pn
    pn = poisson.rvs(pn)    # agrega ruido poissoniano
    pn[pn < 1] = 1
    pn = -np.log(pn / i0)
    pn[pn < 0] = 0  # clipeo los valores negativos

    y = pn  # Sinograma de la imagen
    x0 = np.zeros_like(x)
    x_art = ART(A, AT, y, x0, mu, niter, bpos, plot)
    nor = np.amax(x)

    # data_range = max((x / nor).max(), (x_art / nor).max()) - min((x / nor).min(), (x_art / nor).min())
    data_range = 1.0
    mse_x_art = mse(x / nor, x_art / nor)
    psnr_x_art = psnr(x / nor, x_art / nor, data_range=data_range)
    ssim_x_art = ssim(x_art / nor, x / nor, data_range=data_range)

    return mse_x_art, psnr_x_art, ssim_x_art

def worker(args):
    x, view, detc, noise, niter = args
    mse_x_art, psnr_x_art, ssim_x_art = get_errors(x, view, detc, noise, niter)
    return mse_x_art, psnr_x_art, ssim_x_art