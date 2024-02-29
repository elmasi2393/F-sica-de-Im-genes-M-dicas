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

    L = abs(int(vmax - vmin))
    print("Number of Levels: {}".format(L))

    fig =plt.figure(figsize=(8, 6))
    plt.grid()
    # plt.axis('off')


    imgplot = plt.imshow(im,cmap='gray', vmin=vmin, vmax=vmax)
    fig.colorbar(imgplot, ax=plt.gca())

    plt.figure(figsize=(8, 6))
    hist, bin_edges = np.histogram(im.ravel(),bins=L)
    plt.bar(bin_edges[:-1], hist)


def narest_neighbour_interpolation(im, scale):
    # Obtengo las dimensiones de la imagen vieja y nueva
    old_shape = im.shape
    new_shape = (old_shape[0]*scale, old_shape[1]*scale)

    dx = (old_shape[0]-1)/(new_shape[0]-1)
    dy = (old_shape[1]-1)/(new_shape[1]-1)

    new_im = np.zeros(new_shape)

    # Recorro la imagen nueva
    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            # Calculo la posicion en la imagen vieja (Es casteo se hace hacia arriba si es >0.5 y hacia abajo si es menor)
            old_i = round(i*dx)    
            old_j = round(j*dy)

            # Copio el pixel de la imagen vieja a la nueva
            new_im[i,j] = im[old_i,old_j]

    return new_im

def bilineal_interpolation(im, scale):
    # Obtengo las dimensiones de la imagen vieja y nueva
    old_shape = im.shape
    new_shape = (old_shape[0]*scale, old_shape[1]*scale)
    
    # Creo una imagen nueva con las dimensiones deseadas
    new_im = np.zeros(new_shape)

    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            #calculo la posicion en la imagen vieja
            x = i/(new_shape[0]-1)  #coordenadas relativas
            y = j/(new_shape[1]-1)

            x_int = int(x*(old_shape[0]-1)) #coordenadas enteras
            y_int = int(y*(old_shape[1]-1))

            # Chequeo si no estoy en un borde
            if x_int == old_shape[0]-1 and y_int == old_shape[1]-1:
                new_im[i, j] = im[x_int, y_int]
            elif x_int == old_shape[0]-1:
                new_im[i, j] = int(im[x_int, y_int]*(1 - y) + im[x_int, y_int+1]*y)
            elif y_int == old_shape[1]-1:
                new_im[i, j] = int(im[x_int, y_int]*(1 - x) + im[x_int+1, y_int]*x)
            else:   # Calculo la nueva intensidad
                new_im[i, j] = int(im[x_int, y_int]*(1 - x -y + x*y) + im[x_int+1, y_int]*(x - x*y) + im[x_int, y_int+1]*(y - x*y) + im[x_int+1, y_int+1]*x*y)
    
    return new_im

if __name__ == "__main__":
    
    if(len(sys.argv)<3):
        print("Usage: python read-write-pgm.py [infile.pgm] [outfile.pgm]")
        exit(1)

    infile = sys.argv[1]
    outfiles = sys.argv[2:]
    
    img = read_pgm_file(infile)

    im = np.array(img)
    print("Size of image: {}".format(im.shape))

    show_img_hist(im)

    imout_1 = narest_neighbour_interpolation(im, 8)
    imout_2 = bilineal_interpolation(im, 8)
    imout_3 = cv2.resize(im, (0,0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
    print("Size of image 1: {}".format(imout_1.shape))
    print("Size of image 2: {}".format(imout_2.shape))
    print("Size of image 3: {}".format(imout_3.shape))
    
    show_img_hist(imout_1)
    show_img_hist(imout_2)
    show_img_hist(imout_3)

    
    plt.show()

    cv2.imwrite(outfiles[0],imout_1,[cv2.IMWRITE_PXM_BINARY,0])
    cv2.imwrite(outfiles[1],imout_2,[cv2.IMWRITE_PXM_BINARY,0])
    cv2.imwrite(outfiles[2],imout_3,[cv2.IMWRITE_PXM_BINARY,0])
