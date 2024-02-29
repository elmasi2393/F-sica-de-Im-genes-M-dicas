import cv2
import sys 
import numpy as np
import matplotlib.pyplot as plt
import os.path



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
    fig = plt.figure(figsize=(16,6))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    # imgplot = plt.imshow(im/np.amax(im))
    imgplot = ax1.imshow(im,cmap='gray', vmin=vmin, vmax=vmax)
    fig.colorbar(imgplot, ax=ax1)
    # cv2.imshow(infile,img)
    # cv2.waitKey(0)

    hist, bin_edges = np.histogram(im.ravel(),bins=L)
    ax2.bar(bin_edges[:-1], hist)
    # plt.show()


def noisserer(im, level):
    imout = np.zeros_like(im)
    # Aplico el ruido
    cv2.randn(imout,0,1)
    cv2.add(im,imout*level,imout)
    # imout = np.clip(imout*level+im,0,255).astype(np.uint8)

    # show_img_hist(imout)
    
    # noise = level* np.random.normal(0, 1, im.shape)
    # imout = np.clip(im+noise,0,255)
    # show_img_hist(noise)
    # show_img_hist(imout)
    
    return imout

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
    return cv2.subtract(im, new_image)
    # return im - new_image
def high_boost(im, A, kernel = None):
    if kernel is None:
        kernel = np.array([[1,1,1],[1,1,1],[1,1,1]])*1/9
    new_image = low_pass(im, kernel, method = 'same')
    return cv2.subtract(A*im, new_image)
    # return A*im - new_image
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

    imout = noisserer(im, 10)
    show_img_hist(imout)

    imout = high_boost(imout, 1.0)
    # Reescalear la imagen
    show_img_hist(imout)
    # show_img_hist((imout - imout.min()) * 255/np.amax((imout - imout.min())))

    # imout_2 = cv2.unsharpMask(im, 3, 1.5)
    # show_img_hist(imout_2)
    plt.show()

    

    cv2.imwrite(outfile,imout,[cv2.IMWRITE_PXM_BINARY,0])
