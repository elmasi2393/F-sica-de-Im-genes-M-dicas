import cv2
import sys 
import numpy as np
import matplotlib.pyplot as plt
import os.path
import scienceplots

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

    L = vmax - vmin
    print("Number of Levels: {}".format(L))

    fig =plt.figure(figsize=(8, 6))
    plt.grid()
    plt.axis('off')


    imgplot = plt.imshow(im,cmap='gray', vmin=vmin, vmax=vmax)
    fig.colorbar(imgplot, ax=plt.gca())

    plt.figure(figsize=(8, 6))
    hist, bin_edges = np.histogram(im.ravel(),bins=L)
    plt.bar(bin_edges[:-1], hist)

# Implementa un contrast stretching
def process_pgm_file(im):
    imout = im.copy()

    # Busco los valores maximos y minimos de la imagen
    i_min = np.amin(im)
    i_max = np.amax(im)

    #Aplico la transformacion del contraste T(A) = a * A + b, donde ajusto una funcion lineal para que el valor minimo sea 0 y el maximo 255
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            imout[i,j] = (im[i, j] - i_min) * (255 / (i_max - i_min))
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

    show_img_hist(im)

    imout = process_pgm_file(im)

    show_img_hist(imout)

    plt.show()
    cv2.imwrite(outfile,imout,[cv2.IMWRITE_PXM_BINARY,0])
