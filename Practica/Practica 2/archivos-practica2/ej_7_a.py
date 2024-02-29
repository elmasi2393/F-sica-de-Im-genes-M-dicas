import cv2
import sys 
import numpy as np
import matplotlib.pyplot as plt
import os.path
import pandas as pd
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
    plt.show()

# Permite marcar la ROI utilizada para el analisis de las imagenes
def select_roi(im):
    imout = im.copy()
    
    # Hace una funcion que me permita dibujar un circulo en la imagen que imprima el espectro de magnitud
    def draw_circle(event, x, y, flags, param):
        nonlocal circle_center, is_drawing, radius, image_copy

        if event == cv2.EVENT_LBUTTONDOWN:
            circle_center = (x, y)
            is_drawing = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if is_drawing:
                radius = int(np.sqrt((x - circle_center[0]) ** 2 + (y - circle_center[1]) ** 2))
                image_copy = np.copy(im_3C)
                cv2.circle(image_copy, circle_center, radius, (0, 255, 0), 1)
        elif event == cv2.EVENT_LBUTTONUP:
            is_drawing = False


    cv2.namedWindow("Circle Drawing")
    cv2.setMouseCallback("Circle Drawing", draw_circle)
    
    # Inicializar variables
    circle_center = (-1, -1)
    radius = 0
    is_drawing = False

    im_norm = cv2.normalize(imout, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)   # Normalizo imagen para mostarla
    im_3C = cv2.merge([im_norm, im_norm, im_norm])    # Creo una imagen a color para imprimir el circulo
    image_copy = np.copy(im_3C)   # Copia de la imagen para dibujar el circulo
    

    while True:
        cv2.imshow('Circle Drawing', image_copy)
        # Si presiono enter, guarda el circulo y lo imprime en magnitude_image
        if cv2.waitKey(1) & 0xFF == 13:
            if circle_center[0] != -1 and radius != 0:
                print(f"Circle drawn in {circle_center} with radius {radius}")
                cv2.circle(image_copy, circle_center, radius, (0, 255, 0), 1)
                break
            else:
                print("No circle drawn")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

    plt.axis('off')
    plt.imshow(image_copy)
    plt.show()
    return (circle_center, radius)

def SNR(img, roi, hist = False):
    # Relleno con 0 el circulo en la imagen
    mask = np.zeros(img.shape, dtype=np.float64)
    cv2.circle(mask, roi[0], roi[1], 1, -1)
    pixels = np.ravel(img[mask==1])
    if hist:    # Elejir si plotear el histograma de
        plt.figure()
        plt.hist(pixels, bins=pixels.max()-pixels.min(), alpha=0.5, color='b')
    return pixels.mean()/pixels.std()

if __name__ == "__main__":
    
    # if(len(sys.argv)<3):
    #     print("Usage: python read-write-pgm.py [infile.pgm] [outfile.pgm]")
    #     exit(1)

    infiles = sys.argv[1:]
    
    images = np.array([np.array(read_pgm_file(file)) for file in infiles])

    plt.axis('off')
    plt.imshow(images[0], cmap='gray')
    plt.show()

    print("Size of image: {}".format(images.shape))

    # Marco la ROI en una imagen
    roi = select_roi(images[0])
    print(roi)

    # Calculo el SNR de cada images
    snr = []
    for i in range(len(images)):
        snr.append(SNR(images[i], roi))
        print(f"SNR {infiles[i]}: {snr[i]}")

    plt.show()
    snr = np.array(snr)
    currents = np.array([180, 100, 60])

    plt.plot(currents, snr, 'r-o')
    plt.xlabel('Coriente [mA]')
    plt.ylabel('SNR')
    plt.show()

