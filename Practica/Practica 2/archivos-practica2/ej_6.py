import cv2
import sys 
import numpy as np
import matplotlib.pyplot as plt
import os.path
from scipy.signal import convolve2d
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


def noisserer(im, level):
    imout = np.zeros_like(im)
    # Aplico el ruido
    # cv2.randn(imout,0,1)
    # cv2.add(im,imout*level,imout)
    # imout = np.clip(imout*level+im,0,255).astype(np.uint8)

    # show_img_hist(imout)
    
    noise = level* np.random.normal(0, 1, im.shape)
    imout = im+noise
    
    return cv2.normalize(imout, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_64F)

def low_pass(im, kernel, method = 'valid'):
    # Obtengo las dimensiones de la imagen
    shape = im.shape
    # Obtengo las dimensiones del filtro
    filter_shape = kernel.shape
    #Verifico el method
    if method == 'valid':
        new_shape = (shape[0]-filter_shape[0], shape[1]-filter_shape[1])
        new_im = np.zeros(new_shape)
        # Recorro la imagen nueva
        for i in range(new_shape[0]):
            for j in range(new_shape[1]):
                # Recorro el filtro
                for k in range(filter_shape[0]):
                    for l in range(filter_shape[1]):
                        new_im[i,j] += im[i+k,j+l]*kernel[k,l]
        return new_im
    elif method == 'same':
        new_shape = shape
        # Agrego el pading necesario
        new_im = np.zeros(new_shape)
        pad_x = int(filter_shape[0]/2)
        pad_y = int(filter_shape[1]/2)
        im = np.pad(im,((pad_x,pad_x),(pad_y,pad_y)),'constant')
        # Recorro la imagen nueva
        for i in range(new_shape[0]):
            for j in range(new_shape[1]):
                # Recorro el filtro
                for k in range(filter_shape[0]):
                    for l in range(filter_shape[1]):
                        new_im[i,j] += im[i+k,j+l]*kernel[k,l]
        return new_im

    else:
        print("Invalid method")
        return None
    
def unsarp(im, kernel = None):
    if kernel is None:
        kernel = np.array([[1,1,1],[1,1,1],[1,1,1]])*1/9
    new_image = low_pass(im, kernel, method = 'same')
    # return cv2.subtract(im, new_image)
    return im - new_image

def high_boost(im, A, kernel = None):
    if kernel is None:
        kernel = np.array([[1,1,1],[1,1,1],[1,1,1]])*1/9
    # new_image = low_pass(im, kernel, method = 'same')
    new_image = convolve2d(im, kernel, mode='same')
    # show_img_hist(new_image)
    # return cv2.subtract(A*im, new_image)
    return A*im - new_image
#Calcula el valor absoluto de la diferencia entre 2 imagenes
def distance(x, y):
    return np.sum(np.abs(x-y))
    
if __name__ == "__main__":
    
    if(len(sys.argv)<3):
        print("Usage: python read-write-pgm.py [infile.pgm] [outfile.pgm]")
        exit(1)

    infile = sys.argv[1]
    outfile = sys.argv[2:]
    
    img = read_pgm_file(infile)

    im = np.array(img)
    print("Size of image: {}".format(im.shape))

    show_img_hist(im)
    plt.show()

    imout = noisserer(im, 10)
    show_img_hist(imout)
    plt.show()

    imout_1 = unsarp(imout)
    imout_2 = high_boost(imout, 1.5)
    imout_3 = high_boost(imout, 2)
    imout_4 = high_boost(imout, 8)

    show_img_hist(imout_1)
    show_img_hist(imout_2)
    show_img_hist(imout_3)
    show_img_hist(imout_4)
    plt.show()

    print("Distance 1: ", distance(imout_1, im))
    print("Distance 2: ", distance(imout_2, im))
    print("Distance 3: ", distance(imout_3, im))

    level = np.linspace(0.1,10, 1000)
    d = []
    for i in level:
        imout = high_boost(im, i)
        d.append(distance(imout, im))
    d = np.array(d)
    d = d/d.max()
    peaks, _ = find_peaks(-d)
    print(level[peaks])
    plt.figure()
    plt.plot(level, d)
    plt.plot(level[peaks], np.array(d)[peaks], "rx", ms=15, mew=2)
    plt.xlabel("A")
    plt.ylabel(r"$d/d_{max}$")
    plt.show()

    

    cv2.imwrite(outfile[0],imout_1,[cv2.IMWRITE_PXM_BINARY,0])
    cv2.imwrite(outfile[1],imout_3,[cv2.IMWRITE_PXM_BINARY,0])

