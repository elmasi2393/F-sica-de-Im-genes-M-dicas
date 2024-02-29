import cv2
import sys 
import numpy as np
import matplotlib.pyplot as plt
import os.path
import scienceplots
from scipy.signal import find_peaks

plt.style.use(['science', 'notebook', 'grid'])


def read_pgm_file(file_name):

    data_dir = os.path.dirname(os.path.abspath(__file__))

    # Test if file exists
    file_path = os.path.join(data_dir, file_name)
    assert os.path.isfile(file_path), 'file \'{0}\' does not exist'.format(file_path)

    img = cv2.imread(file_name,flags=cv2.IMREAD_GRAYSCALE)

    if img is not None:
        print('img.size: ', img.size)
    else:
        print('imread({0}) -> None'.format(file_path))

    return img


def show_img_hist(im):
    vmin = np.amin(im)
    vmax = np.max(im)
    print("Intensity Min: {}   Max:{}".format(vmin,vmax))

    L = abs(int(vmax - vmin))
    print("Number of Levels: {}".format(L))

    fig =plt.figure(figsize=(8, 6))
    plt.grid()
    plt.axis('off')


    imgplot = plt.imshow(im,cmap='gray', vmin=vmin, vmax=vmax)
    fig.colorbar(imgplot, ax=plt.gca())

    plt.figure(figsize=(8, 6))
    hist, bin_edges = np.histogram(im.ravel(),bins=L)
    plt.bar(bin_edges[:-1], hist)


def FWHM(im):
    # Imagen original
    plt.figure()
    plt.axis('off')
    plt.imshow(im, cmap='gray')

    y_min = 120
    y_max = 180
    x_min = 190
    x_max = 250
    imout = im[y_min:y_max, x_min:x_max]   # selecciono la franja
    s = 147

    #Franja seleccionada y linea para serializacion
    plt.figure()
    plt.axis('off')
    plt.imshow(imout, cmap='gray')
    plt.axhline(y=s - y_min, color='r', linestyle='--', alpha=0.8)

    plt.show()

    # show_img_hist(imout)

    # Por inspeccion, seleeciono la fila 148:
    # 6, 6, 6
    serializado = im[s,x_min:x_max]
    plt.figure()
    plt.plot(serializado)
    plt.show()

    # Busco el maximo
    peak, _ = find_peaks(serializado, height=100, distance=len(serializado)//2, prominence=100)

    print("Peak: ", peak, serializado[peak])

    # Busco donde el valor de la señal es igual que el maximo/2
    shift = 10
    half_l = np.where(serializado[peak[0]-shift:peak[0]] < serializado[peak]/2)[0]
    half_r = np.where(serializado[peak[0]:peak[0]+shift] < serializado[peak]/2)[0]
    # print("Half: ", half, serializado[half], type(half))

    i_min = peak[0]-shift+half_l[-1]
    i_max = peak[0]+half_r[0]

    half = np.array([i_min, i_max])

    print(f"FWHM: {i_max - i_min}")
    # Muestro el resultado
    plt.figure()
    plt.plot(serializado, '-o', label='señal')
    plt.plot(peak, serializado[peak], "rx", ms = 10, mew = 2, label='máximo')
    plt.axhline(y=serializado[peak]/2, color='g', linestyle='--', alpha=0.6)
    plt.plot(half, serializado[half], "gx", ms=10, mew = 2, label='mitad')
    plt.legend(fontsize=12)
    plt.show()

    return imout

if __name__ == "__main__":
    
    if(len(sys.argv)<3):
        print("Usage: python read-write-pgm.py [infile.pgm] [outfile.pgm]")
        exit(1)

    infile = sys.argv[1]
    outfile = sys.argv[2]
    
    img = read_pgm_file(infile)

    im = np.array(img)
    print("Size of image: {}".format(im.shape))
    # show_img_hist(im)

    imout = FWHM(im)
   

    # plt.figure(figsize=(8, 6))
    # plt.plot(serializado)

    # show_img_hist(imout)
    # plt.show()


    # cv2.imwrite(outfile,imout,[cv2.IMWRITE_PXM_BINARY,0])
