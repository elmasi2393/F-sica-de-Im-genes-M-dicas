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
    plt.axis('off')


    imgplot = plt.imshow(im,cmap='gray', vmin=vmin, vmax=vmax)
    fig.colorbar(imgplot, ax=plt.gca())

    plt.figure(figsize=(8, 6))
    hist, bin_edges = np.histogram(im.ravel(),bins=L)
    plt.bar(bin_edges[:-1], hist)


# Implementa transformacion de thresholding
def thresholding(im):
    imout = im.copy()
    # Aplico el thresholding
    cv2.threshold(im, 127, 255, cv2.THRESH_BINARY_INV, imout)
    
    return imout
# Implamenta transformacion gamma
def gamma(im, g):
    imout = im.copy()
    # Aplico la transformacion gamma
    imout = imout.astype(np.float64)
    imout = np.power(imout, g)
    cv2.normalize(imout, imout, 0, 255, cv2.NORM_MINMAX)
    # c = 255.0/np.amax(imout)
    # imout = c * imout
    return imout


if __name__ == "__main__":
    
    if(len(sys.argv)<3):
        print("Usage: python read-write-pgm.py [infile.pgm] [outfile.pgm]")
        exit(1)

    infile = sys.argv[1]
    outfiles = sys.argv[2:]
    
    img = read_pgm_file(infile)

    im = np.array(img)
    print("Size of image: {}".format(im.shape))

    # show_img_hist(im)

    imout = thresholding(im)
    imout_1 = gamma(im, 0.2)
    imout_2 = gamma(im, 0.5)
    imout_3 = gamma(im, 0.8)
    imout_4 = gamma(im, 1.5)
    imout_5 = gamma(im, 3)


    images = [imout, imout_1, imout_2, imout_3, imout_4, imout_5]
    subs = [im - imout ,im - imout_1, im - imout_5]

    for i in range(len(images)):
        print("Imagen: ", i)
        show_img_hist(images[i])
        plt.show()


    for i in range(len(subs)):
        print("Substraccion: ", i)
        show_img_hist(subs[i])
        plt.show()

    # Guardo las imagenes
    for i in range(len(outfiles)):
        cv2.imwrite(outfiles[i],images[i],[cv2.IMWRITE_PXM_BINARY,0])
    
    # Guardo las imagenes de substraccion (Para ello, poner que todas se subtraigan en el vector subs)
    # for i in range(len(outfiles)):
    #     cv2.imwrite(outfiles[i].split('.')[0] + '_sub.pgm',subs[i],[cv2.IMWRITE_PXM_BINARY,0])