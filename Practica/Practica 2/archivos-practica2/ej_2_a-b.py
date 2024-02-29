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

    L = int(vmax - vmin)
    print("Number of Levels: {}".format(L))

    fig =plt.figure(figsize=(8, 6))
    plt.grid()
    plt.axis('off')

    imgplot = plt.imshow(im,cmap='gray', vmin=vmin, vmax=vmax)
    fig.colorbar(imgplot, ax=plt.gca())

    plt.figure(figsize=(8, 6))
    hist, bin_edges = np.histogram(im.ravel(),bins=L)
    plt.bar(bin_edges[:-1], hist)

# Ecualizacion de histograma
def equalizer(im):
    # Armo el histograma
    hist, _ = np.histogram(im.ravel(),bins=256) # 256 bins (desde 0 a 255)
    # Armo mi primitiva y normalizo
    c = 255.0/(im.shape[0]*im.shape[1]) # constante de normalizacion c = 255/(M*N)
    print("c: ",c)
    print("M*N: ",im.shape[0]*im.shape[1])
    
    primitive = np.zeros(256, dtype=np.float64)
    primitive[0] = c*hist[0]
    # print(primitive[0])
    for i in range(1,len(hist)):    #Itero en las demas
        primitive[i] = primitive[i-1] + c*hist[i]

    # Aplico la transformacion
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            im[i,j] = primitive[im[i,j]]
    
    imout = im
    
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

    imout = equalizer(im)

    show_img_hist(imout)

    plt.show()
    cv2.imwrite(outfile,imout,[cv2.IMWRITE_PXM_BINARY,0])
