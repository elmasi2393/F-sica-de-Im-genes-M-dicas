import cv2
import sys 
import numpy as np
import matplotlib.pyplot as plt
import os.path
from scipy.signal import convolve2d
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
    


def low_pass(im, filter_matrix, method = 'valid'):
    # Obtengo las dimensiones de la imagen
    shape = im.shape
    # Obtengo las dimensiones del filtro
    filter_shape = filter_matrix.shape
    
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
                        new_im[i,j] += im[i+k,j+l]*filter_matrix[k,l]
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
                        new_im[i,j] += im[i+k,j+l]*filter_matrix[k,l]
        return new_im

    else:
        print("Invalid method")
        return None

    


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
    
    filter_matrix_3 = np.ones((3,3))/9
    filter_matrix_5 = np.ones((5,5))/25
    filter_matrix_7 = np.ones((7,7))/49

    mod = 'same'
    imout_3 = low_pass(im, filter_matrix_3, mod)
    imout_5 = low_pass(im, filter_matrix_5, mod)
    imout_7 = low_pass(im, filter_matrix_7, mod)

    # Verifico funcionamiento con funcion de scipy
    # imout_3 = convolve2d(im,filter_matrix_3,mode=mod)
    # imout_5 = convolve2d(im, filter_matrix_5,mode=mod)
    # imout_7 = convolve2d(im, filter_matrix_7,mode=mod)

    print("Filtros aplicados")
    print("Size of image 3x3: {}".format(imout_3.shape))
    print("Size of image 5x5: {}".format(imout_5.shape))
    print("Size of image 7x7: {}".format(imout_7.shape))

    show_img_hist(imout_3)
    show_img_hist(imout_5)
    show_img_hist(imout_7)

    plt.show()

    cv2.imwrite(outfile[1],imout_3,[cv2.IMWRITE_PXM_BINARY,0])
    cv2.imwrite(outfile[2],imout_5,[cv2.IMWRITE_PXM_BINARY,0])
    cv2.imwrite(outfile[3],imout_7,[cv2.IMWRITE_PXM_BINARY,0])
    