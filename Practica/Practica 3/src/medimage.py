import numpy as np
import os
import cv2

# Ecualizacion de histograma
def equalizer(im):

    # Armo el histograma
    hist, _ = np.histogram(im.ravel(),bins=256) # 256 bins (desde 0 a 255)
    # Armo mi primitiva y normalizo al final
    imout = im.copy()
    
    primitive = np.zeros(256, dtype=np.float64)
    primitive[0] = hist[0]
    for i in range(1,len(hist)):    #Itero en las demas
        primitive[i] = primitive[i-1] + hist[i]
    #normalizo
    cv2.normalize(primitive, primitive, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)
    # Aplico la transformacion
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            imout[i,j] = primitive[int(im[i,j])]

    return imout

# Leer archivos .pgm
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